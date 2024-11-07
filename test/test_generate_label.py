from src.generate_dummy_data import generate_data
from src.generate_label import create_image_label


# TODO: tests made for discrete, not for unit circle direction
# def test_label_direction():
#     label = (1, 1, 1)
#     data = generate_data(4, 4, 4, *label)
#     labels = create_image_label(data, label)
#     assert labels[label].tolist() == [0] * 3
#     assert labels[0, 0, 0].tolist() == [1, 1, 1]
#     assert labels[2, 2, 2].tolist() == [-1, -1, -1]
#     assert labels[0, 1, 3].tolist() == [1, 0, -1]


# def test_label_direction_non_cube():
#     label = (0, 3, 1)
#     data = generate_data(4, 8, 2, *label)
#     labels = create_image_label(data, label)
#     assert labels[label].tolist() == [0] * 3
#     assert labels[1, 7, 1].tolist() == [-1, -1, 0]
#     assert labels[2, 2, 1].tolist() == [-1, 1, 0]
#     assert labels[0, 1, 0].tolist() == [0, 1, 1]
