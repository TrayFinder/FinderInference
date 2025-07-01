# 🧠 FinderAI Inference API

This repository provides a **Python service** that enables **real-time product detection and similarity search** using deep learning. It is designed to receive images from a front-end, detect objects, extract embeddings, and return the most visually similar item from a reference database.

---

## 📌 Project Overview

The API:

* Accepts an **image in BLOB/base64 format** via an HTTP endpoint
* Runs **YOLOv11** to **detect the product** in the image
* Crops and processes the detection using **OpenCV**
* Extracts an **embedding vector** using a trained neural network
* Compares it against stored embeddings and returns the **most similar item**

---

## 🚀 How to Run the Project

Follow the steps in [FinderMain](https://github.com/TrayFinder/FinderMain)
---

## ⚙️ Tech Stack

* **FastAPI** – High-performance web framework
* **YOLOv11** – Object detection
* **ONNX Runtime** – Efficient inference for embeddings
* **OpenCV** – Image preprocessing
* **NumPy** – Similarity comparison

## 👥 Authors

This repository is part of the **TrayFinder**, developed by students at \[your university name], in collaboration with **Tray**.
