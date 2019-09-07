#ライブラリ
from PIL import Image,ImageFilter
import os,glob
import numpy as np
from sklearn import model_selection
#リスト作成、数取得、画像ピクセル定義
classes=["miku1","rem","chino"]
num_classes=len(classes)
image_size=50
#検証用データ数定義
num_testdata=15
#画像読み込み
X_train=[]
X_test=[]
Y_train=[]
Y_test=[]
#リスト取り出し、番号を振ってfor文に渡す
for index,classlabel in enumerate(classes):
    #写真のディレクトリ
    photos_dir="./"+classlabel
    #パターン一致でファイル一覧を取得
    files=glob.glob(photos_dir + "/*.jpg")
    #番号を振って順番に画像取り出す
    for i, file in enumerate(files):
        #RGB変換リサイズnumpy変換
        if i>= 147: break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        #検証用データ確保
        if i<num_testdata:
            X_test.append(data)
            Y_test.append(index)
        #画像増幅
        else:
            for angle in range(-20,20,5):
                #回転して追加
                img_r =image.rotate(angle)
                data=np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)
                #反転して追加
                img_trans=img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data=np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)
                #フィルター加工して追加
                img_resample=img_trans.rotate(0,expand=True,resample=Image.BICUBIC)
                data=np.asarray(img_resample)
                X_train.append(data)
                Y_train.append(index)
                #エッジ強化して追加
                img_shr=img_trans.filter(ImageFilter.EDGE_ENHANCE)
                data=np.asarray(img_shr)
                X_train.append(data)
                Y_train.append(index)

#numpy変換
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(Y_train)
y_test=np.array(Y_test)
#xyに格納
xy = (X_train, X_test, y_train, y_test)
#保存
np.save("./anime_aug.npy",xy)
