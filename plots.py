import matplotlib.pyplot as plt
from PIL import Image

# regression_sample'ı import et (PNG'leri otomatik oluşturur)
import regression_sample

# PNG dosyalarını yükle
regression_img = Image.open("regression_plot.png")
debug_img = Image.open("debug_plot.png")

# İki resmi yan yana göster
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Regression plot
axes[0].imshow(regression_img)
axes[0].axis('off')
axes[0].set_title('Regression Plot', fontsize=14, fontweight='bold')

# Debug plot
axes[1].imshow(debug_img)
axes[1].axis('off')
axes[1].set_title('Debug Plot', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("combined.png")

print("PNG dosyaları başarıyla yüklendi ve gösterildi!")