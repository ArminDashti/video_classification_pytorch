from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
#%%
vid, _, _ = read_video("c:/users/armin/desktop/video.mp4", output_format="TCHW")
# vid = vid[:32]  # optionally shorten duration

weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.eval()

preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(vid).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
label = prediction.argmax().item()
score = prediction[label].item()
category_name = weights.meta["categories"][label]
print(f"{category_name}: {100 * score}%")