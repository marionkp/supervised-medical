from src.medical_loader import str_to_landmarks


def test_str_to_landmarks():
    lanmark_str = """
    72, 81, 95
    72, 76, 98
    72, 89, 83
    72, 77, 87
    """
    landmarks = str_to_landmarks(lanmark_str)
    assert landmarks == [
        [72, 81, 95],
        [72, 76, 98],
        [72, 89, 83],
        [72, 77, 87],
    ]
