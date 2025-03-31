import torch, cv2
from torchvision import transforms
from train_classifier import model, DEVICE

transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

def predict(image_path):
    img = cv2.imread(image_path)[:,:,::-1]
    img = transform(img).unsqueeze(0).to(DEVICE)
    pred = torch.argmax(model(img),1).item()
    print(f"Predicted action: {model.fc.out_features[pred]}")

if __name__=='__main__': predict('test.jpg')