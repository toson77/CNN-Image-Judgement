#ライブラリ
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dense,Dropout,Flatten
from keras.utils import np_utils
import keras
import numpy as np
#リスト作成、数取得、画像を50pxと定義
classes=["monkey","boar","crow"]
num_classes=len(classes)
image_size=50

#メイン関数定義
def main():
    #ファイルからテータを配列に読み込む(Xが画像、Yがラベル)
    X_train,X_test,y_train,y_test=np.load("./animal_aug.npy")
    #正規化(0から1の範囲に収める)
    X_train=X_train.astype("float")/256
    X_test=X_test.astype("float")/256
    #正解に1、他0を入れる行列に変換(one-hot-vetcor)
    y_train =np_utils.to_categorical(y_train,num_classes)
    y_test=np_utils.to_categorical(y_test,num_classes)
    #モデル作成関数呼び出し
    model =model_train(X_train,y_train)
    #モデル評価関数呼び出し
    model_eval(model,X_test,y_test)
#CNN定義
def model_train(X,y):
    #連続なモデルであると定義。
    model = Sequential()
    #畳み込み演算。32が出力の次元。3*3のフィルタを用いる。畳み込み結果が同じサイズになるようにする。
    model.add(Conv2D(32,(3,3),padding='same',input_shape=X.shape[1:]))
    #活性化関数(ランプ関数)正の値は通し、負は0
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    #一番大きい値を取り出す
    model.add(MaxPooling2D(pool_size=(2,2)))
    #データ35%捨てる(過学習防止)
    model.add(Dropout(0.35))
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.35))
    #Flatten処理(データを1列に並べる)
    model.add(Flatten())
    #全結合型ニューラルネットワーク
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    #softmax関数(負の値を正の値に変換して確率に変換)
    model.add(Activation('softmax'))
    #最適化アルゴリズム(1回回すごとに学習率を下げていくらしい)
    opt=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
    #損失関数(正解と推定値の誤差)定義、最適化アルゴリズム定義、評価の値を正答率で定義
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    #学習分割数、学習回数定義
    model.fit(X,y, batch_size=128,nb_epoch=40)
    #学習データ保存
    model.save('./animal_cnn_aug.h5')
    return  model
#評価関数定義
def model_eval(model,X, y):
    #結果受け取り
    scores=model.evaluate(X,y,verbose=1)
    #損失量表示
    print('Test Loss:',scores[0])
    #精度表示
    print('Test Accuracy:', scores[1])
#ほかのプログラムから参照する際、このプログラムが呼ばれたときのみmainを実行
if __name__ =="__main__":
    main()
