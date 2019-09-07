#ライブラリ
import os
from flask import Flask,request,redirect,url_for,render_template
from werkzeug.utils import secure_filename
from keras.models import Sequential,load_model
import keras,sys
import numpy as np
from PIL import Image
import tensorflow as tf

#動物の学習データ取り込み
model=load_model('./animal_cnn.h5')
graph = tf.get_default_graph()
classes=["猿","イノシシ","カラス"]
num_classes=len(classes)
image_size=50
#キャラクター学習データ取り込み
model_anime=load_model('./anime_cnn.h5')
graph_anime=tf.get_default_graph()
classes_anime=["ミク","レム","チノ"]
num_classes_anime=len(classes_anime)
image_size_anime=50
#犬猫学習データ取り込み
model_dogcat=load_model('./animal_cnn_dogcat.h5')
graph_dogcat=tf.get_default_graph()
classes_dogcat=["犬","猫"]
num_classes_dogcat=len(classes_dogcat)
image_size_dogcat=50
#アップロードフォルダパス指定
UPLOAD_FOLDER='./static/uploads'
ALLOWED_EXTENSIONS=set(['png','jpg','gif'])

#flaskのおまじない
app=Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
#何かしらの文字入れないと動きませんでした。多分ログイン機能追加するときに使う？
app.config["SECRET_KEY"]="sample"
#.があるかどうかのチェックと、拡張子の確認。
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

#ルーティング
@app.route('/')
def index():
    flag="main"
    #ページ転送
    return render_template('index.html',flag=flag)

#動物画像投稿時のアクション
@app.route('/',methods=['GET','POST'])
def post():
    #学習データ読み込み
    global graph
    with graph.as_default():
     if request.method=='POST':
         #ファイルがなかった時の処理
        if 'file' not in request.files:
            flag="animal_none"
            return render_template('index.html',flag=flag)
        #アップロードされたファイル取り出し
        file =request.files['file']
        #ファイル名がなかった時の処理
        if file.filename == '':
            flag="animal_none"
            return render_template('index.html',flag=flag)
        #アップロードされたファイルのチェック
        if file and allowed_file(file.filename):
            #危険な文字の削除(サニタイズ処理)
            filename = secure_filename(file.filename)
            #ファイル保存
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)

            #RGB変換、リサイズ、numpy配列変換
            image=Image.open(filepath)
            image=image.convert('RGB')
            image=image.resize((image_size,image_size))
            data=np.asarray(image)
            #リスト型変数宣言、data付加、numpy配列変換
            X=[]
            X.append(data)
            X=np.array(X)
            #結果を格納
            result=model.predict([X])[0]
            #最大値を格納
            predicted=result.argmax()
            #百分率化
            percentage=int(result[predicted]*100)
            return render_template('index.html',classes=classes[predicted],percentages=str(percentage),filepath=filepath)

#ルーティング
@app.route('/bunken')
def bunken():
    flag="bunken"
    return render_template("index.html",flag=flag)
#ルーティング
@app.route('/anime')
def ani():
    flag="anime"
    return render_template('index.html',flag=flag)
#キャラクター画像投稿時のアクション(変数が変わるだけで動物画像時と仕組みは同じ)
@app.route('/anime',methods=['GET','POST'])
def anime():
    global graph_anime
    with graph_anime.as_default():
     if request.method=='POST':
        if 'file' not in request.files:
            flag="anime_none"
            return render_template('index.html',flag=flag)
        file =request.files['file']
        if file.filename == '':
            flag="anime_none"
            return render_template('index.html',flag=flag)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)

            image=Image.open(filepath)
            image=image.convert('RGB')
            image=image.resize((image_size_anime,image_size_anime))
            data=np.asarray(image)
            X=[]
            X.append(data)
            X=np.array(X)

            result=model.predict([X])[0]
            predicted=result.argmax()
            percentage_anime=int(result[predicted]*100)
            return render_template('index.html',classes=classes_anime[predicted],percentages_anime=str(percentage_anime),filepath=filepath)
#ルーティング
@app.route('/dogcat')
def dogcat():
    flag="dogcat"
    return render_template('index.html',flag=flag)
#犬猫画像投稿時のアクション(変数が変わるだけで動物画像時と仕組みは同じ)
@app.route('/dogcat',methods=['GET','POST'])
def dogcats():
    global graph_dogcat
    with graph_dogcat.as_default():
     if request.method=='POST':
        if 'file' not in request.files:
            flag="dogcat_none"
            return render_template('index.html',flag=flag)
        file =request.files['file']
        if file.filename == '':
            flag="dogcat_none"
            return render_template('index.html',flag=flag)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)

            image=Image.open(filepath)
            image=image.convert('RGB')
            image=image.resize((image_size_dogcat,image_size_dogcat))
            data=np.asarray(image)
            X=[]
            X.append(data)
            X=np.array(X)

            result=model.predict([X])[0]
            predicted=result.argmax()
            percentage_dogcat=int(result[predicted]*100)
            return render_template('index.html',classes=classes_dogcat[predicted],percentages_dogcat=str(percentage_dogcat),filepath=filepath)

    #最初にimportするとなぜか動かないので直前に書きました。
    from flask import send_flom_directory
    #保存したアップロードファイル呼び出し
    @app.route('/static/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
