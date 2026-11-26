import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


wavelength = 632.8e-9  #red
N = 2048                # Resolução
pixel_size = 8e-6 
M_levels = [2, 4] #níveis de quantização
img = Image.open('CHARIZARD.png').convert('RGB') #abre e converte a imagem de png para rgb
img = img.resize((N, N)) #redimensiona a imagem para a grade de resolução
img_np = np.array(img) / 255.0 #transfere a imagem para uma array NxN e normaliza com valor maximo RGB
I_object = img_np[:, :, 0] # toma apenas o canal Vermelho (R)
np.random.seed(42) #gera uma semente de número aleatório
phase_rand = 2*np.pi*np.random.rand(N,N) #gera N² números aleatórios
U_obj = np.sqrt(I_object)*np.exp(1j*phase_rand) #distribuição de campo do objeto com fase aleatória a partir da raiz da intensidade
U_holo = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_obj))) #excitação do holograma pelo objeto através da fft
phi_holo = np.angle(U_holo)

fig, axes = plt.subplots(2, len(M_levels), figsize=(30, 12)) #reconstruções
for idx, M in enumerate(M_levels):
    phi_norm = np.mod(phi_holo, 2*np.pi) #coloca a fase do holograma no intervalo [0, 2pi[
    phi_quant = np.floor(phi_norm/(2*np.pi/M))*(2*np.pi/M) #discretiza a fase do holograma
    U_h_quant = np.exp(1j*phi_quant) #distribuição do holograma discretizada com base nas fases obtidas pela excitação via objeto
    U_rec = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_h_quant))) #reconstrução do objeto a partir da distribuição do holograma
    I_rec = np.abs(U_rec)**2 #intensidade da reconstrução
    eta= (np.sinc(1 / M))**2 * 100 #eficiência
    axes[0, idx].imshow(phi_quant, cmap='hsv')
    axes[0, idx].set_title(f'Holograma ({M} Níveis)')
    axes[0, idx].axis('off')
    axes[1, idx].imshow(I_rec, cmap='hot')
    axes[1, idx].set_title(f'Reconstrução\n$\eta_{{teor}}$ = {eta:.1f}%')
    axes[1, idx].axis('off')
plt.tight_layout()
plt.show()


