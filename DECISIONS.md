## Những thứ học đc

- Chọn ONNX thay vì Pytorch để có thể deploy lên edge device
- Setup: Docker 512MB/0.5CPU, 50 requests, 5 workers, image size: 256
- Kết quả: All 200 OK, pass test 50 requests và ko crash. Có thể deploy lên Rasberry Pi 4Gb RAM trở lên.

### Runtime và Deployment Optimization

- Update webapp, và Dockerfile để nó ko dùng Torch trong scripts nữa.
- Cụ thể: Dùng python3.12-slim, bỏ torch, chỉ định cụ thể folders và files cần thiết trong Dockerfile, dùng onnx thay thế, chỉnh .dockerignore kĩ càng hơn. Trong data drift, thay vì dùng torch-resnet50 để lọc data thì dùng bản resnet50 ONNX.
- Kết quả: giảm 92% độ nặng của Docker Image, từ 16.7Gb xuống chỉ còn 1.22Gb

### Microservices (Modular Monorepo)

- Thêm 2 folder mới là packages/ và services/
- Trong packages/ có core/ và các packages như ingestion/, transformation/, ... để chúng có thể đc sử dụng mà code ko bị chồng chéo
- Dependencies Management: Mỗi service cài những gì nó cần trong pyproject.toml. Việc cài đặt nội bộ qua pip install -e giúp cô lập môi trường nhưng vẫn giữ được tính independency.
- Deployment Strategy: Mỗi stage là một Docker image độc lập "nhí". Cho phép scale-up riêng lẻ các service nặng (Training/Transformation) và tối ưu hóa tài nguyên (chỉ cấp GPU cho service cần thiết).
- Volume mount chuẩn: ./artifacts:/app/artifacts.
- Mục đích: Tách nhỏ hợp lý có thể tối ưu chi phí (GPU usage) cho service thật sự cần. Ngoài ra, quy trình CICD diễn ra nhanh chóng mà ko gây ra rủi ro cho hệ thống.
