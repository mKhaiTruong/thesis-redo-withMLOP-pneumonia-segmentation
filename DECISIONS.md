## Những thứ học đc

- Chọn ONNX thay vì Pytorch để có thể deploy lên edge device
- Setup: Docker 512MB/0.5CPU, 50 requests, 5 workers, image size: 256
- Kết quả: All 200 OK, pass test 50 requests và ko crash. Có thể deploy lên Rasberry Pi 4Gb RAM trở lên.

### Runtime và Deployment Optimization

- Update webapp, và Dockerfile để nó ko dùng Torch trong scripts nữa.
- Cụ thể: Dùng python3.12-slim, bỏ torch, chỉ định cụ thể folders và files cần thiết trong Dockerfile, dùng onnx thay thế, chỉnh .dockerignore kĩ càng hơn. Trong data drift, thay vì dùng torch-resnet50 để lọc data thì dùng bản resnet50 ONNX.
- Kết quả: giảm 92% độ nặng của Docker Image, từ 16.7Gb xuống chỉ còn 1.22Gb
