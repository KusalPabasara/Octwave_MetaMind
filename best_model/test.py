import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

CONFIG = {
    'data_dir': os.environ.get('DATA_DIR', 'data/data'),
    'test_csv': os.environ.get('TEST_CSV', 'test.csv'),
    'model_path': os.environ.get('MODEL_PATH', 'resnet50_f1_optimized.pth'),
    'label_encoders_path': os.environ.get('LABEL_ENCODERS', 'label_encoders_resnet50.pth'),
    'submission_path': os.environ.get('SUBMISSION_PATH', 'submission_resnet50_HIGH_RISK_PLUS.csv'),
    'batch_size': int(os.environ.get('BATCH_SIZE', '16')),
    'image_size': int(os.environ.get('IMAGE_SIZE', '224')),
    'num_workers': int(os.environ.get('NUM_WORKERS', '4')),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # Thresholds: HIGH_RISK_PLUS that achieved 0.58661
    'threshold_added': float(os.environ.get('THRESH_ADDED', '0.48')),
    'threshold_removed': float(os.environ.get('THRESH_REMOVED', '0.68')),
    'threshold_changed': float(os.environ.get('THRESH_CHANGED', '0.70')),
}

class SpotDifferenceTestDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['img_id']
        img1_path = os.path.join(self.data_dir, f"{img_id}_1.png")
        img2_path = os.path.join(self.data_dir, f"{img_id}_2.png")
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        combined = torch.cat([img1, img2], dim=0)
        return combined, img_id

class SpotDifferenceModel(nn.Module):
    def __init__(self, num_classes_added, num_classes_removed, num_classes_changed):
        super().__init__()
        backbone = models.resnet50(pretrained=False)
        backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc_added = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes_added)
        )
        self.fc_removed = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes_removed)
        )
        self.fc_changed = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes_changed)
        )
    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        return self.fc_added(f), self.fc_removed(f), self.fc_changed(f)

@torch.no_grad()
def predict_with_thresholds(outputs_added, outputs_removed, outputs_changed,
                           mlb_added, mlb_removed, mlb_changed,
                           thr_added, thr_removed, thr_changed):
    preds = []
    for i in range(len(outputs_added)):
        pa = torch.sigmoid(outputs_added[i]).cpu().numpy()
        pr = torch.sigmoid(outputs_removed[i]).cpu().numpy()
        pc = torch.sigmoid(outputs_changed[i]).cpu().numpy()
        ia = np.where(pa >= thr_added)[0]
        ir = np.where(pr >= thr_removed)[0]
        ic = np.where(pc >= thr_changed)[0]
        added = ' '.join(mlb_added.classes_[ia]) if len(ia) else 'none'
        removed = ' '.join(mlb_removed.classes_[ir]) if len(ir) else 'none'
        changed = ' '.join(mlb_changed.classes_[ic]) if len(ic) else 'none'
        preds.append({'added_objs': added, 'removed_objs': removed, 'changed_objs': changed})
    return preds

def main():
    print('='*70)
    print('Running best-so-far inference (HIGH_RISK_PLUS thresholds)')
    print('='*70)
    print('Device:', CONFIG['device'])
    test_df = pd.read_csv(CONFIG['test_csv'])
    print('Test samples:', len(test_df))

    label_encoders = torch.load(CONFIG['label_encoders_path'], weights_only=False)
    mlb_added = label_encoders['added']
    mlb_removed = label_encoders['removed']
    mlb_changed = label_encoders['changed']

    model = SpotDifferenceModel(len(mlb_added.classes_), len(mlb_removed.classes_), len(mlb_changed.classes_))
    print('Loading model from', CONFIG['model_path'])
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device'], weights_only=True))
    model = model.to(CONFIG['device']).eval()

    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = SpotDifferenceTestDataset(test_df, CONFIG['data_dir'], transform)
    dl = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    all_preds, all_ids = [], []
    for imgs, ids in tqdm(dl, desc='Predicting'):
        imgs = imgs.to(CONFIG['device'])
        oa, or_, oc = model(imgs)
        batch = predict_with_thresholds(oa, or_, oc, mlb_added, mlb_removed, mlb_changed,
                                        CONFIG['threshold_added'], CONFIG['threshold_removed'], CONFIG['threshold_changed'])
        all_preds.extend(batch)
        all_ids.extend(ids)

    img_ids_clean = [int(x) if torch.is_tensor(x) else x for x in all_ids]
    sub = pd.DataFrame({
        'img_id': img_ids_clean,
        'added_objs': [p['added_objs'] for p in all_preds],
        'removed_objs': [p['removed_objs'] for p in all_preds],
        'changed_objs': [p['changed_objs'] for p in all_preds],
    })
    sub.to_csv(CONFIG['submission_path'], index=False)

    # Print stats
    total = len(sub)
    a_none = (sub['added_objs']=='none').sum(); r_none=(sub['removed_objs']=='none').sum(); c_none=(sub['changed_objs']=='none').sum()
    print('\nSaved:', CONFIG['submission_path'])
    print('Added none:', f"{100*a_none/total:.1f}%", 'Removed none:', f"{100*r_none/total:.1f}%", 'Changed none:', f"{100*c_none/total:.1f}%")

if __name__ == '__main__':
    main()
