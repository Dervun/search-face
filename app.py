import argparse
from base64 import b64encode
import io
import os
import pickle
from typing import List
from urllib.parse import quote

from flask import Flask, request
from flask_cors import CORS
import numpy as np
from PIL import Image

from big_brother import BigBrother

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default='data/')
args = parser.parse_args()

app = Flask(__name__)
CORS(app)

if not os.path.isdir(args.datadir):
    os.makedirs(args.datadir)
data_file = os.path.join(args.datadir, 'all_data.pkl')
if os.path.isfile(data_file):
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)
    system = BigBrother(all_data['embeddings'], all_data['names'],
                        all_data['top_k'], all_data['model'], args.datadir)
else:
    system = BigBrother(data_path=args.datadir)


def image_from_list(list_image: List[List[int]]) -> np.ndarray:
    # expected that shape is height, width, channels (3)
    image = np.array(list_image, dtype=np.uint8)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f'image shape: {image.shape}')
    return image


@app.route('/get_nearest', methods=['GET'])
def get_nearest():
    data = request.get_json()  # Get data posted as a json
    # check that all the required parameters have been provided
    expected_variables = ('image',)
    for variable in expected_variables:
        if variable not in data:
            answer = {'success': 0,
                      'message': f'Expected `{variable}` variable'}
            return answer

    try:
        image = image_from_list(data['image'])
    except Exception:
        answer = {
            'success': 0,
            'message': 'Wrong image format, expected: dimensions are [height, '
                       'width, channels], where channels == 3, and values are '
                       'integers from 0 to 255'}
        return answer

    try:
        params = {'image': image}
        if 'top_k' in data:
            params['top_k'] = data['top_k']
        distances, ids, names = system.get_nearest(**params)
        answer = {'success': 1, 'distances': distances, 'ids': ids,
                  'names': names}
    except Exception:
        answer = {'success': 0, 'message': 'Something went wrong'}

    return answer


@app.route('/get_nearest_compressed', methods=['POST', 'GET'])
def get_nearest_compressed():
    if 'image' not in request.files:
        answer = {
            'success': 0,
            'message': 'Expected `image` file, for example: curl -F '
                       '"image=@some_image.png" .../get_nearest_compressed'}
        return answer

    try:
        image_binary = request.files['image'].read()
        pil_image = Image.open(io.BytesIO(image_binary))
        image = np.array(pil_image, dtype=np.uint8)
    except Exception:
        answer = {
            'success': 0,
            'message': 'Wrong image format'}
        return answer

    try:
        params = {'image': image}
        if 'top_k' in request.values:
            params['top_k'] = int(request.values['top_k'])
        distances, ids, names = system.get_nearest(**params)

        answer = {'success': 1, 'distances': distances, 'ids': ids,
                  'names': names}
    except Exception:
        answer = {'success': 0, 'message': 'Something went wrong'}

    return answer


@app.route('/get_images_by_id', methods=['GET'])
def get_images_by_id():
    data = request.get_json()  # Get data posted as a json
    # check that all the required parameters have been provided
    expected_variables = ('id',)
    for variable in expected_variables:
        if variable not in data:
            answer = {'success': 0,
                      'message': f'Expected `{variable}` variable'}
            return answer

    try:
        ids = data['id']
        if isinstance(ids, int):
            ids = [ids]

        # TODO: change to URL to S3
        images_folder = system.get_images_folder()
        images = []
        for i in ids:
            current_path = os.path.join(images_folder, f'{i}.png')
            images.append(current_path)
        image_format = data.get('format', 'raw')
        if image_format == 'raw':
            result = [np.array(Image.open(p), dtype=np.uint8).tolist()
                      for p in images]
        elif image_format == 'base64':
            result = []
            for p in images:
                with open(p, 'rb') as f:
                    image_binary = f.read()
                data = b64encode(image_binary).decode('ascii')
                result.append(f'data:image/png;base64,{quote(data)}')
        else:
            answer = {'success': 0,
                      'message': 'Unknown format, expected "raw", "base64"'}
            return answer
        answer = {'success': 1, 'images': result}

    except Exception:
        answer = {'success': 0, 'message': 'Something went wrong'}

    return answer


@app.route('/add_face_photo', methods=['POST'])
def add_face_photo():
    data = request.get_json()  # Get data posted as a json
    # check that all the required parameters have been provided
    # TODO: change name to unique ID
    expected_variables = ('image', 'name')
    for variable in expected_variables:
        if variable not in data:
            answer = {'success': 0,
                      'message': f'Expected `{variable}` variable'}
            return answer

    try:
        image = image_from_list(data['image'])
    except Exception:
        answer = {
            'success': 0,
            'message': 'Wrong image format, expected: dimensions are [height, '
                       'width, channels], where channels == 3, and values are '
                       'integers from 0 to 255'}
        return answer

    try:
        name = data['name']
        system.add_new_face(image, name)

        # TODO: change writing to disk to writing to some DB
        with open(data_file, 'wb') as f:
            new_dump = system.get_all_data()
            pickle.dump(new_dump, f)
        answer = {'success': 1, 'message': 'All right :)'}
    except Exception:
        answer = {'success': 0, 'message': 'Something went wrong'}

    return answer


@app.route('/add_face_photo_compressed', methods=['POST'])
def add_face_photo_compressed():
    if 'image' not in request.files or 'name' not in request.values:
        answer = {
            'success': 0,
            'message': 'Expected `image` file and `name` parameter, for '
                       'example: curl -F "image=@some_image.png" -F '
                       '"name=First Last" .../add_face_photo_compressed'}
        return answer

    try:
        image_binary = request.files['image'].read()
        pil_image = Image.open(io.BytesIO(image_binary))
        image = np.array(pil_image, dtype=np.uint8)
    except Exception:
        answer = {
            'success': 0,
            'message': 'Wrong image format'}
        return answer

    try:
        name = request.values['name']
        system.add_new_face(image, name)

        # TODO: change writing to disk to writing to some DB
        with open(data_file, 'wb') as f:
            new_dump = system.get_all_data()
            pickle.dump(new_dump, f)
        answer = {'success': 1, 'message': 'All right :)'}
    except Exception:
        answer = {'success': 0, 'message': 'Something went wrong'}

    return answer


def main():
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    # app.run(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
