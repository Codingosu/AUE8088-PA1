from PIL import Image

# 이미지 열기
img = Image.open('datasets/tiny-imagenet-200/test/images/test_0.JPEG')

# (width, height) 출력
print("이미지 크기 (width, height):", img.size)