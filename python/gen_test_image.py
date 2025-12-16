import torch
from torchvision import datasets, transforms
import os

def main():
    # 1. Setari
    data_dir = "./data"
    
    # 2. Incarcam datele de test MNIST
    # Nu folosim DataLoader, ci direct Dataset-ul pentru a lua o singura imagine
    tr = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tr)
    
    # 3. Alegem o imagine (de exemplu indexul 0, care este cifra 7)
    img, label = test_ds[0] 
    
    # Imaginea e un tensor [1, 28, 28]. O aplatizam la [784]
    img_data = img.numpy().flatten()
    
    # 4. Cream folderul hls daca nu exista
    os.makedirs("hls", exist_ok=True)
    
    # 5. Scriem fisierul header C++
    header_path = "hls/test_image.h"
    print(f"Generating {header_path} for label: {label}...")
    
    with open(header_path, "w") as f:
        f.write(f"// Imagine generata din MNIST Test set. Label corect: {label}\n")
        f.write("#ifndef TEST_IMAGE_H\n")
        f.write("#define TEST_IMAGE_H\n\n")
        
        f.write("const float test_image[784] = {\n")
        
        for i, val in enumerate(img_data):
            # Scriem valoarea (float intre 0.0 si 1.0)
            f.write(f"{val:.6f}, ")
            # Formatare: linie noua la fiecare 15 valori ca sa arate frumos
            if (i + 1) % 15 == 0: 
                f.write("\n")
        
        f.write("\n};\n\n")
        
        # Scriem si label-ul asteptat ca sa il putem verifica in Testbench
        f.write(f"const int expected_label = {label};\n")
        
        f.write("#endif\n")
        
    print("Done! Acum poti rula simularea HLS.")

if __name__ == "__main__":
    main()