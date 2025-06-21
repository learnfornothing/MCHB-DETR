from ultralytics import YOLO
from ultralytics import RTDETR

if __name__ == '__main__':
    # 直接使用预训练模型创建模型.
    # model = YOLO('yolov8n.pt')
    # model.train(**{'cfg': 'ultralytics/cfg/exp1.yaml', 'data': 'dataset/data.yaml'})

    # 使用yaml配置文件来创建模型,并导入预训练权重.
    model = RTDETR('ultralytics/cfg/models/rtdetr+MC-Backbone+HiLoAIFI+BScaseq.yaml')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml')
    model.train(
                cfg=r"/root/yolov8-main/ultralytics/cfg/default.yaml",
                data=r"/root/yolov8-main/ultralytics/datasets/inkjet.yaml",
                # cfg=r"D:\root\yolov8-main\ultralytics\cfg\default.yaml",
                # data=r"D:\root\yolov8-main\ultralytics\datasets\inkjet.yaml",
                imgsz=640,
                epochs=250,
                batch=16,
                workers=8,
                device=0,
                project='runs/inkjet/train',
                optimizer='Adam',
                name=' ',
                patience=100,
                )

    # 模型验证
    # model = RTDETR(r'/root/autodl-tmp/proiect/yolov8-main/runs/Cxray/rtdetr+MC-Backbone+HiLoAIFI/train/weights/best.pt')
    # model = YOLO('/root/autodl-tmp/proiect/yolov8-main/runs/yolov5m/train2/weights/best.pt')
    # model.val(data=r'/root/autodl-tmp/proiect/yolov8-main/ultralytics/datasets/inkjet.yaml',
    #           project='/runs/XXX-DETR',
    #           name='val')

    # model = RTDETR(r"D:\root\yolov8-main\runs\Cxray\rtdetr+MC-Backbone+HiLoAIFI\train\weights\best.pt")
    # model = YOLO(r"D:\root\yolov8-main\runs\yolov10m\train2\weights\best.pt")
    # model.val(
    #     data=r'D:\root\yolov8-main\ultralytics\datasets\inkjet.yaml',
    #     project='/runs/rtdetr+MELAN(SE)',
    #     name='val'
    # )


    # 模型推理
    # model = RTDETR(r'D:\root\yolov8-main\runs\rtdetr+MC-Backbone+HiLoAIFI\train3\weights\best.pt')
    # # model = YOLO(r'D:\Dataset\MCHB-DETR\Cxray\YOLOv8m\train\weights\best.pt')
    # model.predict(
    #               # source='ultralytics/datasets/Cxray/test/images',
    #               source='ultralytics/datasets/Inkjetdata/test/images/2.bmp',
    #               # **{'save': True},
    #               save=True,
    #               project='D:\Dataset\MCHB-DETR\Inkejet\MCHB-DETR\Visualize',
    #               name='predict',
    #               visualize=True
    #               )
