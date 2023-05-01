from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


@app.route('/')
def temp():
    return render_template('template.html')

@app.route('/',methods=['POST','GET'])
def get_input():
    if request.method == 'POST':
        info = request.form['search']
        return redirect(url_for('run_pred',values=info))

@app.route('/run_pred/<values>')
def run_pred(values):
    import numpy as np 
    import tensorflow as tf
    from keras.utils import load_img, img_to_array
    from PIL import Image
    
    model = tf.keras.models.load_model('model18.h5')
    
    values = load_img(values, target_size=(256,256))
    values = img_to_array(values)
    values = np.array([values])
    pred = int(model.predict(values)[0][0].round())
    
    if pred == 0:
        return 'No abnormalities detected'
    return 'Abnormalities detected'
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)