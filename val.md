# 这一章节我们的目的是验证效果，因为B60作为一款消费级的显卡，他的作用是替代某卡的时候，消费者为什么要选择这个卡呢？

| 指标                      | PyTorch（xpu） | OpenVINO（GPU.0） | 差异              |
| ----------------------- | ------------ | --------------- | --------------- |
| **Precision (P)**       | 0.612        | 0.622           | ↑ +0.010        |
| **Recall (R)**          | 0.406        | 0.393           | ↓ -0.013        |
| **mAP50**               | 0.471        | 0.445           | ↓ -0.026        |
| **mAP50-95**            | 0.324        | 0.311           | ↓ -0.013        |
| **推理时间 Inference time** | **43.3 ms**  | **2.2 ms**      | **🔥 19.7× 加速** |
| **总耗时（val benchmark）**  | ~6.8 s       | ~0.8 s          | **🔥 8.5× 加速**  |



 - 先使用原生的来跑一下
   
```bash
(B60) root@b60:~# yolo val model=/root/ultralytics/runs/detect/train4/weights/best.pt data=coco128.yaml device=xpu imgsz=256
Model summary (fused): 72 layers, 3,151,904 parameters, 0 gradients, 8.7 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2567.3±1347.5 MB/s, size: 46.3 KB)
val: Scanning /root/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 128/128 3.3Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 8/8 1.2it/s 6.8s
                   all        128        929      0.612      0.406      0.471      0.324
                person         61        254       0.59      0.508      0.538      0.359
               bicycle          3          6      0.361      0.167      0.167      0.117
                   car         12         46      0.626      0.087     0.0936      0.057
            motorcycle          4          5      0.431          1      0.938      0.732
              airplane          5          6          1      0.816      0.871      0.732
                   bus          5          7      0.753      0.571      0.644      0.578
                 train          3          3      0.643          1      0.913      0.789
                 truck          5         12      0.427     0.0833      0.233      0.124
                  boat          2          6      0.449      0.333      0.485      0.342
         traffic light          4         14      0.709      0.143       0.21      0.131
             stop sign          2          2      0.495        0.5      0.745      0.497
                 bench          5          9      0.283      0.111      0.191      0.125
                  bird          2         16      0.653      0.473      0.595      0.187
                   cat          4          4      0.613          1      0.945      0.673
                   dog          9          9      0.588      0.667       0.71      0.556
                 horse          1          2      0.723          1      0.995      0.514
              elephant          4         17      0.992      0.588      0.723      0.449
                  bear          1          1      0.577          1      0.995      0.995
                 zebra          2          4      0.831          1      0.995      0.995
               giraffe          4          9      0.666      0.778      0.852      0.638
              backpack          4          6      0.832      0.333      0.343       0.12
              umbrella          4         18      0.644      0.444      0.544      0.245
               handbag          9         19          0          0    0.00567    0.00293
                   tie          6          7      0.688      0.286      0.391      0.292
              suitcase          2          4      0.463       0.75      0.707      0.454
               frisbee          5          5      0.765        0.6      0.605      0.525
                  skis          1          1          0          0     0.0585    0.00585
             snowboard          2          7          1      0.638      0.859       0.51
           sports ball          6          6          0          0     0.0219     0.0197
                  kite          2         10       0.27     0.0541     0.0913     0.0246
          baseball bat          4          4          1          0          0          0
        baseball glove          4          7      0.839      0.286      0.292      0.148
            skateboard          3          5      0.543        0.2      0.199      0.139
         tennis racket          5          7      0.385      0.286       0.29       0.21
                bottle          6         18      0.828      0.269      0.244      0.175
            wine glass          5         16      0.558      0.162      0.163     0.0931
                   cup         10         36      0.636      0.222      0.246      0.171
                  fork          6          6      0.355      0.167      0.198      0.164
                 knife          7         16      0.187       0.25      0.148      0.114
                 spoon          5         22          1          0     0.0922     0.0357
                  bowl          9         28      0.666      0.393       0.48      0.328
                banana          1          1          1          0      0.995      0.239
              sandwich          2          2          0          0      0.284      0.284
                orange          1          4          1          0      0.129     0.0739
              broccoli          4         11      0.248      0.182      0.191      0.179
                carrot          3         24      0.796      0.164       0.34      0.258
               hot dog          1          2      0.581          1      0.663      0.614
                 pizza          5          5      0.554          1      0.839      0.728
                 donut          2         14      0.633      0.929      0.911      0.752
                  cake          4          4      0.368       0.75      0.628      0.511
                 chair          9         35      0.424        0.4      0.305      0.121
                 couch          5          6      0.436      0.393      0.542      0.255
          potted plant          9         14      0.882      0.214      0.322      0.217
                   bed          3          3          1      0.655      0.863      0.577
          dining table         10         13      0.388      0.634      0.492      0.381
                toilet          2          2      0.531        0.5      0.501      0.501
                    tv          2          2      0.301        0.5      0.586      0.469
                laptop          2          3          1          0      0.106     0.0574
                 mouse          2          2          1          0          0          0
                remote          5          8      0.773      0.375       0.44      0.336
            cell phone          5          8          1          0          0          0
             microwave          3          3      0.414      0.667      0.624      0.446
                  oven          5          5      0.418        0.4      0.317      0.271
                  sink          4          6      0.497      0.333      0.187      0.125
          refrigerator          5          5      0.721      0.531      0.683      0.479
                  book          6         29          1          0      0.123     0.0389
                 clock          8          9      0.712      0.778      0.827      0.523
                  vase          2          2      0.288        0.5      0.662      0.612
              scissors          1          1          1          0      0.995     0.0995
            teddy bear          6         21      0.708      0.286       0.49       0.29
            toothbrush          2          5      0.694      0.465      0.564      0.233
Speed: 0.1ms preprocess, 43.3ms inference, 0.0ms loss, 1.7ms postprocess per image
Results saved to /root/ultralytics/runs/detect/val4
💡 Learn more at https://docs.ultralytics.com/modes/val
VS Code: view Ultralytics VS Code Extension ⚡ at https://docs.ultralytics.com/integrations/vscode
````

 - 再使用部署后的结果跑一下呗

```bash
(B60) root@b60:~# yolo val model=/root/ultralytics/runs/detect/train4/weights/best_openvino_model data=coco128.yaml device=openvino:GPU.0 imgsz=256
Loading /root/ultralytics/runs/detect/train4/weights/best_openvino_model for OpenVINO inference...
Using OpenVINO LATENCY mode for batch=1 inference on (CPU)...
Setting batch=1 input of shape (1, 3, 256, 256)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1831.3±769.6 MB/s, size: 76.6 KB)
val: Scanning /root/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 128/128 700.0Kit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 128/128 156.0it/s 0.8s
                   all        128        929      0.622      0.393      0.445      0.311
                person         61        254       0.63      0.475      0.521      0.353
               bicycle          3          6      0.362      0.167       0.17      0.152
                   car         12         46      0.472      0.087      0.109     0.0583
            motorcycle          4          5      0.502          1      0.906       0.73
              airplane          5          6          1      0.765      0.866      0.697
                   bus          5          7      0.645      0.714      0.697      0.609
                 train          3          3      0.615          1      0.995      0.929
                 truck          5         12      0.297     0.0833      0.171     0.0847
                  boat          2          6      0.538      0.333       0.44      0.359
         traffic light          4         14      0.692      0.143      0.173      0.117
             stop sign          2          2      0.751        0.5      0.745      0.498
                 bench          5          9      0.433      0.178      0.212      0.149
                  bird          2         16      0.567      0.375      0.588      0.207
                   cat          4          4      0.485       0.75      0.822      0.579
                   dog          9          9        0.6      0.444      0.496      0.391
                 horse          1          2      0.552          1      0.995      0.547
              elephant          4         17      0.914      0.412      0.546      0.383
                  bear          1          1      0.568          1      0.995      0.895
                 zebra          2          4      0.831          1      0.995      0.995
               giraffe          4          9      0.829      0.889      0.888      0.685
              backpack          4          6      0.736      0.333      0.414      0.129
              umbrella          4         18      0.635      0.389      0.528      0.244
               handbag          9         19          0          0     0.0038    0.00243
                   tie          6          7        0.7      0.286      0.396      0.279
              suitcase          2          4       0.56      0.651      0.639       0.43
               frisbee          5          5      0.797        0.6      0.609      0.528
                  skis          1          1      0.197          1      0.249     0.0258
             snowboard          2          7       0.68      0.309      0.618      0.391
           sports ball          6          6          0          0     0.0209     0.0188
                  kite          2         10          1          0     0.0616     0.0192
          baseball bat          4          4          1          0          0          0
        baseball glove          4          7      0.826      0.286      0.292      0.148
            skateboard          3          5      0.588        0.2       0.22      0.152
         tennis racket          5          7      0.247      0.286      0.291      0.172
                bottle          6         18      0.656      0.222      0.229       0.17
            wine glass          5         16      0.575      0.172      0.177     0.0989
                   cup         10         36       0.69      0.248      0.241      0.143
                  fork          6          6      0.502      0.167      0.207      0.167
                 knife          7         16      0.305      0.375      0.294      0.174
                 spoon          5         22          1          0     0.0817     0.0391
                  bowl          9         28      0.841      0.429      0.483      0.325
                banana          1          1          1          0      0.332     0.0332
              sandwich          2          2     0.0595      0.149      0.176      0.176
                orange          1          4          1          0     0.0893     0.0472
              broccoli          4         11      0.344      0.182       0.19       0.17
                carrot          3         24      0.797       0.25       0.34       0.25
               hot dog          1          2      0.534          1      0.663      0.622
                 pizza          5          5      0.407        0.8        0.8       0.67
                 donut          2         14      0.711       0.88      0.866      0.697
                  cake          4          4      0.377       0.75      0.586      0.442
                 chair          9         35      0.373      0.286      0.255      0.118
                 couch          5          6      0.188      0.167      0.234      0.128
          potted plant          9         14      0.932      0.214      0.357      0.198
                   bed          3          3          1          0      0.583      0.339
          dining table         10         13      0.355      0.538      0.433      0.348
                toilet          2          2      0.566        0.5      0.503      0.503
                    tv          2          2      0.604          1      0.995      0.796
                laptop          2          3          1          0     0.0531      0.024
                 mouse          2          2          1          0          0          0
                remote          5          8      0.801      0.375      0.458      0.309
            cell phone          5          8          1          0          0          0
             microwave          3          3      0.213      0.333      0.455      0.366
                  oven          5          5      0.359        0.4      0.378      0.348
                  sink          4          6      0.474      0.333      0.201      0.134
          refrigerator          5          5      0.719        0.6      0.649      0.475
                  book          6         29          1          0      0.113     0.0434
                 clock          8          9      0.557      0.667      0.711      0.543
                  vase          2          2      0.553        0.5      0.828      0.679
              scissors          1          1          1          0      0.995     0.0995
            teddy bear          6         21      0.558      0.333      0.435      0.276
            toothbrush          2          5      0.826        0.4      0.543      0.194
Speed: 0.1ms preprocess, 2.2ms inference, 0.0ms loss, 0.7ms postprocess per image
Results saved to /root/ultralytics/runs/detect/val3
💡 Learn more at https://docs.ultralytics.com/modes/val
VS Code: view Ultralytics VS Code Extension ⚡ at https://docs.ultralytics.com/integrations/vscode
````

未来要做的：拿其它品牌的显卡对比一下
