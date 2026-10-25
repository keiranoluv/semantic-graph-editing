from scripts.cache_vg150_regions import scale_boxes_to_jpeg


def test_gt_box_scales_when_jpeg_size_differs_from_metadata():
    assert scale_boxes_to_jpeg([(0, 0, 1024, 768), (256, 192, 512, 384)], (1024, 768), (1000, 586)) == [
        (0, 0, 1000, 586),
        (250, 146.5, 500, 293),
    ]
