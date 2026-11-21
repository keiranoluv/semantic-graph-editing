"""Composable attribute JSON: train tokens + input-word pointer + ASCII fallback."""
import json
import re

import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, EOS = 0, 1, 2
MAX_STEPS = 192
MAX_MEMORY = 128


def pieces(text):
    return re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]", text)


def serialize(attributes):
    return json.dumps(sorted(attributes), ensure_ascii=False, separators=(",", ":"))


def input_words(source, instruction):
    values = []
    texts = [instruction] + [a for n in source.nodes for a in n.attributes]
    for text in texts:
        for t in pieces(text):
            if t.isalnum() and t not in values:
                values.append(t)
    return tuple(values[:MAX_MEMORY])


def vocabulary(train):
    tokens = {chr(i) for i in range(32, 127)}
    for e in train:
        for g in (e.source, e.target):
            for n in g.nodes:
                tokens.update(pieces(serialize(n.attributes)))
        tokens.update(pieces(e.instruction))
    return ["<pad>", "<bos>", "<eos>", *sorted(tokens)]


def encode(attributes, fixed, memory):
    local = fixed + list(memory)
    index = {v: i for i, v in reversed(list(enumerate(local)))}
    result = []
    for token in pieces(serialize(attributes)):
        if token in index:
            result.append(index[token])
        else:
            if any(c not in index for c in token):
                raise ValueError("Unrepresentable attribute character")
            result.extend(index[c] for c in token)
    if len(result) + 1 > MAX_STEPS:
        raise ValueError("Attribute sequence exceeds fixed length budget")
    return [*result, EOS]


def decode(ids, fixed, memory):
    local = fixed + list(memory)
    if EOS not in ids:
        raise ValueError("Unterminated attribute sequence")
    ids = ids[: ids.index(EOS)]
    if any(i < 3 or i >= len(local) for i in ids):
        raise ValueError("Invalid attribute token")
    values = json.loads("".join(local[i] for i in ids))
    if (
        not isinstance(values, list)
        or len(values) > 32
        or any(not isinstance(v, str) or not v or "=" not in v for v in values)
    ):
        raise ValueError("Invalid typed attribute list")
    if len(values) != len(set(values)):
        raise ValueError("Repeated attribute atom")
    return tuple(sorted(values))


class AttributeDecoder(nn.Module):
    def __init__(self, hidden, tokens, vectors):
        super().__init__()
        self.tokens = tokens
        self.register_buffer("fixed_vectors", vectors.detach().clone())
        self.input = nn.Linear(vectors.shape[-1], hidden)
        self.start = nn.Linear(hidden, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.generate = nn.Linear(hidden, len(tokens))
        self.query = nn.Linear(hidden, vectors.shape[-1])
        self.copy_bias = nn.Parameter(torch.tensor(-1.0))

    def step(self, previous, state, memory, mask, matches):
        fixed = self.fixed_vectors[None].expand(len(state), -1, -1)
        vectors = torch.cat((fixed, memory), 1)
        embedded = vectors.gather(1, previous[:, None, None].expand(-1, 1, vectors.shape[-1])).squeeze(1)
        state = self.cell(self.input(embedded), state)
        generated = self.generate(state)
        copied = (
            10
            * torch.bmm(
                F.normalize(self.query(state), dim=-1)[:, None], F.normalize(memory, dim=-1).transpose(1, 2)
            ).squeeze(1)
            + self.copy_bias
        )
        copied = copied.masked_fill(~mask, -1e4)
        extra = torch.full_like(generated, -torch.inf).scatter(
            -1, matches.clamp_min(0), copied.masked_fill(matches < 3, -torch.inf)
        )
        logits = torch.cat((torch.logaddexp(generated, extra), copied.masked_fill(matches >= 3, -1e4)), -1)
        logits[:, PAD] = -1e4
        logits[:, BOS] = -1e4
        return logits, state

    def teacher_loss(self, context, memory, mask, matches, targets):
        state = torch.tanh(self.start(context))
        previous = torch.full((len(context),), BOS, dtype=torch.long, device=context.device)
        total, count = context.sum() * 0, 0
        for t in range(targets.shape[1]):
            logits, state = self.step(previous, state, memory, mask, matches)
            selected = targets[:, t] != -100
            total = total + F.cross_entropy(logits[selected], targets[selected, t], reduction="sum")
            count += int(selected.sum())
            previous = targets[:, t].clamp_min(0)
        return total / count

    def greedy(self, context, memory, mask, matches):
        state = torch.tanh(self.start(context))
        previous = torch.full((len(context),), BOS, dtype=torch.long, device=context.device)
        finished = torch.zeros(len(context), dtype=torch.bool, device=context.device)
        output = []
        for _ in range(MAX_STEPS):
            logits, state = self.step(previous, state, memory, mask, matches)
            previous = logits.argmax(-1)
            output.append(previous)
            finished |= previous == EOS
            if finished.all():
                break
        return torch.stack(output, 1).cpu().tolist()
