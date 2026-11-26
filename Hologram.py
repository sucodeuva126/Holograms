import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 1. PARÂMETROS FÍSICOS DO SISTEMA
# ============================================================

wavelength = 632.8e-9  # Comprimento de onda (632,8 nm)
N = 2048                # Resolução do holograma (2048 x 2048)
pixel_size = 8e-6      # Tamanho do pixel (8 µm)

# Número de níveis de quantização a serem avaliados
M_levels = [2, 4, 8, 16]


# ============================================================
# 2. CARREGAMENTO E PRÉ-PROCESSAMENTO DA IMAGEM
# ============================================================

# Carregar a imagem e converter para RGB
img = Image.open('CHARIZARD.png').convert('RGB')

# Redimensionar a imagem para 512 x 512 pixels
img = img.resize((N, N))

# Converter a imagem para uma matriz NumPy normalizada
img_np = np.array(img) / 255.0

# Selecionar somente o canal vermelho (R)
I_object = img_np[:, :, 0]


# ============================================================
# 3. GERAÇÃO DA FASE ALEATÓRIA E DO CAMPO NO OBJETO
# ============================================================

# Definir semente para garantir a reprodutibilidade
np.random.seed(42)

# Gerar distribuição de fase aleatória entre 0 e 2π
phase_rand = 2 * np.pi * np.random.rand(N, N)

# Construir o campo complexo no plano do objeto
# Amplitude = raiz quadrada da intensidade
U_object = np.sqrt(I_object) * np.exp(1j * phase_rand)


# ============================================================
# 4. CÁLCULO DO HOLOGRAMA DE FOURIER
# ============================================================

# Aplicar Transformada de Fourier bidimensional
U_holo = np.fft.fftshift(
    np.fft.fft2(
        np.fft.ifftshift(U_object)
    )
)

# Extrair a distribuição de fase do holograma
phi_holo = np.angle(U_holo)


# ============================================================
# 5. QUANTIZAÇÃO DA FASE E RECONSTRUÇÃO
# ============================================================

# Criar figura para apresentar os resultados
fig, axes = plt.subplots(
    2,
    len(M_levels),
    figsize=(15, 6)
)

for idx, M in enumerate(M_levels):

    # --------------------------------------------------------
    # 5.1. Quantização da fase
    # --------------------------------------------------------

    # Ajustar a fase para o intervalo [0, 2π)
    phi_norm = np.mod(phi_holo, 2 * np.pi)

    # Calcular o intervalo angular de cada nível
    delta_phi = 2 * np.pi / M

    # Discretizar a fase em M níveis
    phi_quant = np.floor(
        phi_norm / delta_phi
    ) * delta_phi

    # --------------------------------------------------------
    # 5.2. Campo transmitido pelo holograma de fase
    # --------------------------------------------------------

    # Considerar amplitude unitária e modulação de fase
    U_h_quant = np.exp(1j * phi_quant)

    # --------------------------------------------------------
    # 5.3. Reconstrução por Transformada Inversa de Fourier
    # --------------------------------------------------------

    U_rec = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(U_h_quant)
        )
    )

    # Calcular a intensidade reconstruída
    I_rec = np.abs(U_rec) ** 2

    # --------------------------------------------------------
    # 5.4. Cálculo da eficiência teórica
    # --------------------------------------------------------

    # Eficiência teórica baseada na função sinc²
    eta_teorica = (np.sinc(1 / M)) ** 2 * 100

    # --------------------------------------------------------
    # 5.5. Exibição dos resultados
    # --------------------------------------------------------

    # Exibir holograma de fase quantizado
    axes[0, idx].imshow(
        phi_quant,
        cmap='twilight',
        vmin=0,
        vmax=2 * np.pi
    )

    axes[0, idx].set_title(
        f'Holograma ({M} Níveis)'
    )

    axes[0, idx].axis('off')

    # Exibir imagem reconstruída
    axes[1, idx].imshow(
        I_rec,
        cmap='gray'
    )

    axes[1, idx].set_title(
        f'Reconstrução\n'
        f'η teor. = {eta_teorica:.1f}%'
    )

    axes[1, idx].axis('off')


# ============================================================
# 6. APRESENTAÇÃO FINAL DOS RESULTADOS
# ============================================================

plt.tight_layout()
plt.show()


