from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
import io

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的前端域名，设置为 "*" 表示允许所有
    allow_credentials=True,
    allow_methods=["*"],  # 允许的 HTTP 方法
    allow_headers=["*"],  # 允许的 HTTP 请求头
)

# 西班牙自治区列表
COMUNIDADES = [
    "Andalucía",
    "Aragón",
    "Asturias",
    "Canary Is.",
    "Cantabria",
    "Castilla-La Mancha",
    "Castilla y León",
    "Cataluña",
    "Valenciana",
    "Extremadura",
    "Galicia",
    "Islas Baleares",
    "La Rioja",
    "Madrid",
    "Murcia",
    "País Vasco",
    "Navarra",
    "Ceuta",
    "Melilla",
]

ckpt_path = "geolocal/StreetCLIP"
ckpt_path = "results/checkpoint-1"
model = CLIPModel.from_pretrained(ckpt_path).to(device)
processor = CLIPProcessor.from_pretrained(ckpt_path)


def predict(image_paths, batch_size):
    pridicted_labels = []
    for i in tqdm.tqdm(range(0, len(image_paths), batch_size)):
        images = [
            Image.open(image_path)
            for image_path in image_paths[i : min(i + batch_size, len(image_paths))]
        ]
        inputs = processor(
            text=COMUNIDADES,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model(**inputs)

        for image in images:
            image.close()

        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)
        pridicted_labels += [COMUNIDADES[i] for i in probs.argmax(dim=1).tolist()]

    return pridicted_labels


def on_pic_demo(image):
    inputs = processor(
        text=COMUNIDADES,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs)

    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)

    # for each photo, Show labels ordered by decreasing probability
    return sorted(zip(COMUNIDADES, probs[0].tolist()), key=lambda x: x[1], reverse=True)


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # 确保上传文件是图片类型
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Only JPEG and PNG are supported.",
            )

        # 读取文件内容
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # 调用预测函数
        result = on_pic_demo(image)

        # 返回结果
        return JSONResponse(content={"predictions": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


def get_accuracy(
    labels_json="dataset_spain/comunidad.json",
    images_folder="dataset_spain",
    batch_size=2,
):

    with open(labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)

    image_paths = []
    true_labels = []
    for image_id, true_comunidad in labels.items():
        image_paths.append(os.path.join(images_folder, f"{image_id}.png"))
        true_labels.append(true_comunidad)

    images = [Image.open(image_path) for image_path in image_paths]

    predicted_labels = predict(images, batch_size=batch_size)

    # 计算准确率
    print(
        f"Accuracy: {sum([1 for i, j in zip(true_labels, predicted_labels) if i == j]) / len(true_labels) * 100:.2f}%"
    )


# if __name__ == "__main__":
#     with Image.open("dataset_demo/20241208_143822.jpg") as image:
#         result = on_pic_demo(image)

#     for label, prob in result:
#         print(f"{label}: {prob:.2%}%")
