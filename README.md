## Criando um Sistema de Reconhecimento Facial do Zero.

---

![bairesDev](https://github.com/user-attachments/assets/19a85b48-9a12-43e0-b34f-df1210dd9148)

**Bootcamp BairesDev — Machine Learning Training | Ministrado pela DIO**

---

# 🚀 reconheceFaceZero — Sistema de Reconhecimento Facial

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-red)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-GPU%20Ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> *A diferença entre um projeto e um portfólio é o contexto que você coloca nele.*

---

## 1. 🎯 Problema de Negócio

Sistemas de controle de acesso, segurança e autenticação baseados em senhas ou crachás físicos apresentam dois problemas críticos: são facilmente burlados e não identificam **quem** está presente — apenas se a credencial é válida.

O desafio é construir um sistema capaz de **detectar e identificar rostos em tempo real**, associando cada face detectada a uma identidade conhecida, sem depender de tokens físicos ou senhas. O sistema precisa funcionar tanto em imagens estáticas quanto em vídeos, e ser extensível para novos indivíduos sem retreinamento completo da pipeline.

---

## 2. 🏢 Contexto

O projeto foi desenvolvido no **Bootcamp BairesDev — Machine Learning Training**, como aplicação end-to-end de visão computacional combinando detecção e reconhecimento de faces.

A solução adota uma arquitetura de **dois estágios** deliberadamente desacoplados:

**Estágio 1 — Detecção:** YOLOv8 localiza e recorta todas as faces presentes na imagem ou frame de vídeo, retornando bounding boxes com coordenadas pixel a pixel.

**Estágio 2 — Identificação:** Para cada crop detectado, `face_recognition` (baseado em dlib/ResNet) extrai um vetor de 128 dimensões que representa a geometria facial. Um classificador SVM linear (scikit-learn) mapeia esse vetor ao nome da pessoa correspondente.

O dataset de treinamento é organizado pelo usuário em `data/persons/NOME_PESSOA/`, tornando o sistema adaptável a qualquer conjunto de identidades. A infraestrutura é containerizada com Docker + CUDA 11.8, com um `Makefile` que centraliza todos os comandos operacionais.

---

## 3. 📐 Premissas da Análise

As seguintes premissas delimitam o escopo desta implementação:

- O detector YOLOv8 é treinado especificamente para **detecção de faces** (classe única: `face`), usando anotações no formato YOLO geradas externamente com ferramentas como LabelImg. O `config/data.yaml` define os caminhos de treino/validação e a classe alvo.
- A extração de embeddings usa a biblioteca `face_recognition`, que internamente aplica um modelo ResNet pré-treinado para gerar vetores de **128 dimensões** por face. Esse embedding é invariante a pequenas variações de iluminação e pose.
- O classificador escolhido foi **SVM com kernel linear** (`probability=True`). Para datasets com poucas amostras por pessoa (típico em cenários reais de controle de acesso), SVMs lineares generalizam melhor do que redes neurais completas, que exigiriam muito mais dados.
- Imagens com dimensão lateral maior que 2.000 pixels são sinalizadas como potencialmente problemáticas por `src/data_utils.py`, pois reduzem a velocidade de inferência sem ganho mensurável de precisão nas detecções.
- O pipeline de inferência (`src/inference_pipeline.py`) opera sobre frames BGR (padrão OpenCV) e converte internamente para RGB antes de passar pela extração de embeddings — alinhamento necessário porque `face_recognition` espera RGB.
- Os modelos treinados (`.pt` e `.pkl`) são ignorados pelo `.gitignore` — apenas o código e a estrutura de diretórios são versionados.

---

## 4. 🛠️ Estratégia da Solução

A pipeline foi implementada em cinco etapas incrementais, cada uma mapeada a um notebook Colab dedicado:

**Etapa 1 — Preparação do Dataset (`01_prepare_data_colab`)**
Cria a estrutura de diretórios no Google Drive e auxilia no upload organizado das imagens por pessoa. Cada subpasta em `data/persons/` corresponde a um label de identidade. O notebook também visualiza amostras para confirmar a organização antes do treinamento.

**Etapa 2 — Treinamento do Detector YOLOv8 (`02_train_yolo_colab`)**
Carrega `yolov8n.pt` (nano — mais rápido e viável no Colab gratuito) e fine-tuna com 50 épocas no dataset anotado, usando `imgsz=640` e `batch=16`. O `train.sh` local automatiza esse processo via `src/yolov8_train.py`, que encapsula a API do Ultralytics e retorna o caminho do `best.pt`.

**Etapa 3 — Extração de Embeddings (`03_extract_embeddings_colab`)**
Percorre `data/persons/` pessoa por pessoa, abre cada imagem com `face_recognition.load_image_file()` e extrai `face_encodings()`. Os vetores de 128d são salvos em `models/embeddings.npy` e os labels em `models/labels.npy`, prontos para o treinamento do classificador.

**Etapa 4 — Treinamento do Classificador SVM (`04_train_classifier_colab`)**
Carrega os arrays `.npy`, treina `SVC(kernel='linear', probability=True)` e serializa o modelo com `joblib.dump()` em `models/classifier/svm_face_recog.pkl`. A etapa inteira leva segundos, mesmo com centenas de identidades.

**Etapa 5 — Inferência (`05_inference_demo_colab` e `inference.sh`)**
O `FaceRecognizer` em `src/inference_pipeline.py` carrega ambos os modelos e expõe um único método `recognize(image_bgr)` que retorna lista de `{'box': (x1,y1,x2,y2), 'name': str}`. O `inference.sh` detecta automaticamente se o input é imagem ou vídeo e aplica o pipeline correspondente com renderização em tempo real via OpenCV.

---

## 5. 💡 Insights Técnicos

**Por que dois estágios separados (YOLOv8 + SVM) em vez de um modelo único end-to-end?**
Um modelo face-to-identity end-to-end exigiria centenas de imagens por pessoa para treinamento. A arquitetura em dois estágios resolve isso: o YOLOv8 foi treinado uma vez para detectar faces genericamente; o SVM é retreinado rapidamente cada vez que uma nova identidade é adicionada. Adicionar uma nova pessoa ao sistema custa apenas: tirar fotos, extrair embeddings, retreinar o SVM — sem tocar no detector.

**Por que `face_recognition` (dlib) e não o encoder do próprio YOLOv8 ou FaceNet TensorFlow?**
O `face_recognition` encapsula um modelo ResNet pré-treinado em ~3 milhões de faces, gerando embeddings de 128d com alta robustez. A alternativa FaceNet TF estaria disponível (`src/embeddings.py` documenta essa opção), mas exigiria infraestrutura de GPU para inferência em tempo real. Para o escopo do projeto, dlib oferece precisão comparável com CPU.

**Por que SVM linear e não KNN ou Random Forest?**
Com embeddings de 128d gerados por um modelo de métrica (face_recognition), o espaço vetorial já é semanticamente organizado — faces da mesma pessoa tendem a formar clusters compactos. SVM linear encontra hiperplanos de separação nesse espaço de forma eficiente e com boa generalização mesmo com poucas amostras. KNN seria sensível a outliers; Random Forest fragmentaria desnecessariamente um espaço já bem separado.

**Por que `nvidia/cuda:11.8.0-cudnn8-runtime` como base do Dockerfile e não a imagem oficial do PyTorch?**
A imagem base CUDA mantém o container enxuto. As dependências Python são instaladas via `requirements.txt`, dando controle total sobre as versões. Imagens PyTorch pré-construídas tendem a incluir versões fixas que podem conflitar com `ultralytics` e `face_recognition`. O `docker-compose.yml` usa `runtime: nvidia` e `NVIDIA_VISIBLE_DEVICES=all` para passar a GPU integralmente ao container.

**Por que `make` e não um script shell único?**
O `Makefile` funciona como documentação executável. `make build`, `make up-detach`, `make shell`, `make logs` — qualquer colaborador sabe exatamente o que fazer sem ler o README. É uma decisão de DX (Developer Experience) que comunicada profissionalismo imediatamente ao abrir o repositório.

---

## 6. 📊 Resultados

O projeto entrega uma pipeline de reconhecimento facial funcional e modular com os seguintes artefatos:

| Artefato | Descrição |
|---|---|
| `models/yolo/best.pt` | Detector YOLOv8n fine-tuned para faces |
| `models/embeddings.npy` | Vetores 128d por imagem do dataset |
| `models/labels.npy` | Labels de identidade correspondentes |
| `models/classifier/svm_face_recog.pkl` | Classificador SVM treinado |
| `src/inference_pipeline.py` | `FaceRecognizer` — API unificada de inferência |
| Notebooks 01–05 | Pipeline reprodutível passo a passo no Colab |

O sistema é capaz de:
- Detectar múltiplas faces simultaneamente em imagens e vídeos
- Identificar cada face com nome e bounding box renderizado
- Escalar para novas identidades sem retreinar o detector
- Rodar localmente com GPU via Docker ou no Google Colab gratuitamente

---

## 7. 🔭 Próximos Passos

- Implementar **data augmentation** nas imagens de treino (`albumentations` já está no `environment.yml`) para aumentar robustez a variações de iluminação e ângulo.
- Substituir o classificador SVM por **ArcFace** ou **FaceNet** para aprendizado métrico direto, melhorando a separabilidade com poucos exemplos por pessoa.
- Expor o `FaceRecognizer` como **API REST com FastAPI** (infraestrutura já mapeada no `docker-compose.yml` na porta 5000).
- Adicionar **TensorBoard** para monitoramento do treinamento YOLOv8 (porta 6006 já exposta no `docker-compose.yml`).
- Implementar **threshold de confiança** no SVM: abaixo de um score mínimo, retornar `"Desconhecido"` em vez de forçar uma predição incorreta.

---

## 💻 Requisitos

### Hardware

| Recurso | Mínimo | Recomendado |
|---|---|---|
| CPU | Quad-Core | 8 núcleos |
| RAM | 8 GB | 16 GB |
| GPU | — | NVIDIA CUDA 11.8+ |
| Disco | 20 GB livres | 40 GB livres |

### Software

| Dependência | Versão |
|---|---|
| Python | 3.10 |
| PyTorch | 2.0+ |
| ultralytics (YOLOv8) | 8.0.20+ |
| face_recognition | última estável |
| scikit-learn | última estável |
| Docker + NVIDIA Container Toolkit | para uso de GPU |
| Conda | opcional (ambiente local) |

---

## 📂 Estrutura do Projeto

```bash
reconheceFaceZero/
│── Dockerfile              # nvidia/cuda:11.8-cudnn8-runtime + Python 3.10
│── docker-compose.yml      # GPU passthrough completo — portas 8888/6006/5000
│── Makefile                # build, up, up-detach, down, shell, logs, clean
│── environment.yml         # Conda: PyTorch CUDA 11.8 + albumentations + ultralytics
│── requirements.txt        # pip: YOLOv8, PyTorch, OpenCV, scikit-learn, face_recognition
│── setup.sh                # Cria estrutura de pastas + git init + pip install
│── train.sh                # YOLOv8 fine-tuning → extração embeddings → treino SVM
│── inference.sh            # Inferência em imagem ou vídeo com renderização OpenCV
│── run_jupyter.sh          # Abre Jupyter no container Docker já em execução
│── README.md
│
├── config/
│   └── data.yaml           # Paths treino/val + nc=1 + names=['face']
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py       # list_person_images, check_image_sizes (>2000px)
│   ├── detect_and_crop.py  # YOLOv8 → crops de faces (filtro min_area=1000px²)
│   ├── embeddings.py       # face_recognition → vetores 128d (BGR→RGB interno)
│   ├── train_classifier.py # SVC(kernel='linear', probability=True) + joblib.dump
│   ├── yolov8_train.py     # Wrapper ultralytics.YOLO.train() → retorna best.pt
│   └── inference_pipeline.py # FaceRecognizer: detector + clf → recognize(bgr)
│
├── notebooks/
│   ├── 01_prepare_data_colab.ipynb   # Drive mount + estrutura de pastas
│   ├── 02_train_yolo_colab.ipynb     # YOLOv8n fine-tuning 50 épocas
│   ├── 03_extract_embeddings_colab.ipynb  # face_recognition → .npy
│   ├── 04_train_classifier_colab.ipynb    # SVM → svm_face_recog.pkl
│   └── 05_inference_demo_colab.ipynb      # Pipeline completa em imagem
│
├── data/
│   ├── persons/            # NOME_PESSOA/ → imagens brutas por identidade
│   └── yolo_annotations/
│       ├── images/train/   # Imagens para treino YOLOv8
│       ├── images/val/     # Imagens para validação
│       ├── labels/train/   # Anotações YOLO (.txt) — treino
│       └── labels/val/     # Anotações YOLO (.txt) — validação
│
├── models/
│   ├── yolo/               # best.pt após treinamento (ignorado pelo .gitignore)
│   ├── classifier/         # svm_face_recog.pkl (ignorado pelo .gitignore)
│   ├── embeddings.npy      # Vetores 128d (ignorado pelo .gitignore)
│   └── labels.npy          # Labels de identidade (ignorado pelo .gitignore)
│
└── tests/
    └── test_inference.py   # Valida carregamento do FaceRecognizer com modelos dummy
```

---

## ⚙️ Como Executar

### 🔹 Google Colab (Recomendado para início)

Execute os notebooks na ordem:

```
01_prepare_data_colab → 02_train_yolo_colab → 03_extract_embeddings_colab
→ 04_train_classifier_colab → 05_inference_demo_colab
```

Cada notebook instala suas dependências e persiste modelos no Google Drive.

### 🔹 Localmente com Docker (GPU)

**1. Clonar e configurar:**

```bash
git clone https://github.com/Santosdevbjj/reconheceFaceZero.git
cd reconheceFaceZero
bash setup.sh
```

**2. Build e iniciar o container:**

```bash
make build
make up-detach
```

**3. Treinar a pipeline completa:**

```bash
./train.sh
```

**4. Rodar inferência em uma imagem:**

```bash
./inference.sh caminho/para/imagem.jpg
```

**5. Rodar inferência em vídeo:**

```bash
./inference.sh caminho/para/video.mp4
```

**6. Abrir Jupyter Notebook:**

```bash
make up-detach
bash run_jupyter.sh
# Acesse: http://localhost:8888
```

**7. Acessar o shell do container:**

```bash
make shell
```

**8. Rodar testes:**

```bash
pytest tests/
```

### 🔹 Localmente com Conda (sem Docker)

```bash
conda env create -f environment.yml
conda activate yolo-env-gpu
./train.sh
```

---

## 📖 Notebooks

| Notebook | Objetivo |
|---|---|
| `01_prepare_data_colab` | Estrutura do Drive + upload organizado por pessoa |
| `02_train_yolo_colab` | Fine-tuning YOLOv8n — detecção de faces |
| `03_extract_embeddings_colab` | `face_recognition` → vetores 128d salvos em `.npy` |
| `04_train_classifier_colab` | SVM linear → `svm_face_recog.pkl` |
| `05_inference_demo_colab` | Pipeline completa em imagem com renderização |

---

## 📌 Aprendizados

**1. Dois estágios separados valem mais do que um modelo monolítico.** A decisão de desacoplar detecção (YOLOv8) e identificação (SVM) foi a mais importante do projeto. Significa que adicionar uma nova pessoa ao sistema custa apenas tirar fotos, extrair embeddings e retreinar o SVM — sem tocar no detector. Aprendi que arquitetura modular não é sobre elegância, é sobre custo de manutenção.

**2. O espaço de embeddings do `face_recognition` já é uma métrica.** Quando comecei, pensei em usar KNN por ser simples. Mas entendi que vetores gerados por um modelo treinado com triplet loss ou contrastive loss já formam clusters naturais — o SVM linear encontra hiperplanos de separação nesse espaço de forma muito mais eficiente do que vizinhos mais próximos com ruído de borda.

**3. O `Makefile` é documentação que roda.** Criar atalhos como `make build`, `make shell` e `make logs` não foi trabalho extra — foi economizar tempo meu e de qualquer pessoa que clone o repositório. Um recrutador que abre o README e vê um `Makefile` com comandos claros entende imediatamente que o código foi escrito para ser usado, não apenas lido.

---

## 📜 Licença

Este projeto está sob a licença MIT. Sinta-se livre para usar, modificar e distribuir.

---

## 📬 Contato

[![Portfólio Sérgio Santos](https://img.shields.io/badge/Portfólio-Sérgio_Santos-111827?style=for-the-badge&logo=githubpages&logoColor=00eaff)](https://portfoliosantossergio.vercel.app)
[![LinkedIn Sérgio Santos](https://img.shields.io/badge/LinkedIn-Sérgio_Santos-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/santossergioluiz)

---




