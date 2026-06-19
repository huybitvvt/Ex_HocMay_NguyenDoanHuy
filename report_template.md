# Báo cáo B3/B4 - Optimization với CNN trên CIFAR-10

Link code GitHub: `DIEN_LINK_GITHUB_O_DAY`

## B3. Bảng so sánh

Sau khi chạy các cấu hình CNN trên CIFAR-10, em thu được bảng kết quả như sau. Bảng này dùng để so sánh mức loss cuối, độ chính xác trên tập validation, thời gian train và số epoch mô hình bắt đầu gần đạt kết quả tốt nhất.

| Mô hình | Optimizer | BatchNorm | Dropout | Training loss cuối | Validation accuracy | Thời gian train | Epoch convergence |
|---|---|---:|---:|---:|---:|---:|---:|
| BasicCNN | SGD | Không | 0.0 | ... | ... | ... | ... |
| CNN_BN_Dropout | SGD + Momentum | Có | 0.2 | ... | ... | ... | ... |
| CNN_BN_Dropout | SGD + Momentum | Có | 0.5 | ... | ... | ... | ... |
| CNN_BN_Dropout | Adam | Có | 0.2 | ... | ... | ... | ... |
| CNN_BN_Dropout | Adam | Có | 0.5 | ... | ... | ... | ... |

## B4. Kết luận

Trong các thí nghiệm đã chạy, em chọn mô hình có validation accuracy cao nhất làm mô hình tốt nhất. Ngoài accuracy, em cũng xem thêm training loss và thời gian train để tránh chọn một mô hình chỉ tốt ngẫu nhiên ở một epoch nhưng học không ổn định.

Nhìn chung, việc thêm Batch Normalization giúp quá trình học ổn định hơn so với CNN cơ bản dùng SGD. Dropout có tác dụng giảm overfitting, nhưng nếu đặt quá cao thì mô hình học chậm hơn vì bị loại bớt nhiều thông tin trong quá trình train. Với optimizer, Adam thường cải thiện nhanh trong những epoch đầu, còn SGD + Momentum có thể cần thêm thời gian nhưng vẫn cho kết quả tốt nếu learning rate phù hợp.

Vì vậy, mô hình tốt nhất không chỉ là mô hình có accuracy cao nhất, mà còn là cấu hình có loss giảm đều, validation accuracy ổn định và thời gian train hợp lý.
