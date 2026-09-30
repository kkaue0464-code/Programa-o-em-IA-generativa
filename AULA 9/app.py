import streamlit as st
import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

# Desativa a pausa padrão do pyautogui para evitar latência no loop
pyautogui.PAUSE = 0.0
# Recurso de segurança: mover o mouse para o canto superior esquerdo da tela cancela o script
pyautogui.FAILSAFE = True

# 1. Configuração da Página Streamlit
st.set_page_config(
    page_title="Controle de Mouse por Movimento das Mãos",
    page_icon="🖐️",
    layout="wide"
)

st.title("🖐️ Controle de Mouse por Visão Computacional")
st.markdown("Mova o ponteiro do mouse utilizando o movimento da sua mão (pulso) detectado por **YOLOv8-Pose**.")

# 2. Carregamento do Modelo Pose (Cacheado)
@st.cache_resource
def load_pose_model():
    # Modelo otimizado para estimativa de pose e keypoints em CPU
    return YOLO("yolov8n-pose.pt")

model = load_pose_model()

# 3. Sidebar - Configurações e Controles
st.sidebar.header("Configurações do Controle")

camera_index = st.sidebar.number_input("Índice da Câmera", min_value=0, max_value=5, value=0)

wrist_choice = st.sidebar.radio(
    "Mão/Pulso de Controle",
    options=["Direita", "Esquerda"],
    index=0
)
# No COCO Keypoints: ID 9 = Pulso Esquerdo, ID 10 = Pulso Direito
WRIST_KEYPOINT_ID = 10 if wrist_choice == "Direita" else 9

conf_threshold = st.sidebar.slider(
    "Limiar de Confiança",
    min_value=0.2,
    max_value=1.0,
    value=0.5,
    step=0.05
)

smooth_factor = st.sidebar.slider(
    "Fator de Suavização (EMA)",
    min_value=0.05,
    max_value=0.9,
    value=0.25,
    step=0.05,
    help="Valores menores tornam o mouse mais suave, valores maiores tornam mais rápido."
)

if "mouse_running" not in st.session_state:
    st.session_state.mouse_running = False

col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("▶️ Iniciar Mouse"):
    st.session_state.mouse_running = True
if col_b2.button("⏹ Parar Mouse"):
    st.session_state.mouse_running = False

# Resolução da tela do sistema
screen_w, screen_h = pyautogui.size()

# 4. Layout
frame_window = st.empty()
status_window = st.sidebar.empty()

# 5. Loop de Processamento e Controle
if st.session_state.mouse_running:
    cap = cv2.VideoCapture(int(camera_index))

    if not cap.isOpened():
        st.error(f"Não foi possível acessar a câmera no índice {camera_index}.")
        st.session_state.mouse_running = False
    else:
        prev_x, prev_y = None, None

        try:
            while st.session_state.mouse_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Falha ao capturar quadro da câmera.")
                    break

                # Espelhar horizontalmente a imagem para que o movimento seja natural
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Inferência da Pose em CPU
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    device="cpu",
                    verbose=False
                )[0]

                annotated_frame = frame.copy()
                hand_detected = False

                # Processar Keypoints detectados
                if results.keypoints is not None and len(results.keypoints) > 0:
                    # Pega a primeira pessoa detectada na cena
                    keypoints = results.keypoints[0].data[0]

                    if len(keypoints) > WRIST_KEYPOINT_ID:
                        wrist = keypoints[WRIST_KEYPOINT_ID]
                        x_kp, y_kp, conf_kp = float(wrist[0]), float(wrist[1]), float(wrist[2])

                        if conf_kp >= conf_threshold:
                            hand_detected = True

                            # Desenhar o ponto do pulso no vídeo
                            cv2.circle(annotated_frame, (int(x_kp), int(y_kp)), 10, (0, 255, 0), -1)
                            cv2.putText(
                                annotated_frame,
                                f"Mao ({wrist_choice})",
                                (int(x_kp) + 15, int(y_kp)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )

                            # Normalização e mapeamento para coordenadas de tela
                            target_x = (x_kp / w) * screen_w
                            target_y = (y_kp / h) * screen_h

                            # Aplicar filtro de suavização (Média Móvel Exponencial)
                            if prev_x is None or prev_y is None:
                                curr_x, curr_y = target_x, target_y
                            else:
                                curr_x = smooth_factor * target_x + (1 - smooth_factor) * prev_x
                                curr_y = smooth_factor * target_y + (1 - smooth_factor) * prev_y

                            prev_x, prev_y = curr_x, curr_y

                            # Mover o ponteiro do mouse no SO
                            pyautogui.moveTo(int(curr_x), int(curr_y))

                # Atualizar visualização do vídeo
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

                # Status na sidebar
                if hand_detected:
                    status_window.success(f"**Mão Detectada**\n\nPosição Mouse: ({int(prev_x)}, {int(prev_y)})")
                else:
                    status_window.warning("Procurando mão na câmera...")

        finally:
            cap.release()
            frame_window.empty()
            status_window.empty()
else:
    st.info("Clique em **'▶️ Iniciar Mouse'** para começar a controlar o cursor.")