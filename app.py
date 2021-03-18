import argparse
import os
import pickle
from typing import List

from flask import Flask, request
from flask_cors import CORS
import numpy as np

from big_brother import BigBrother

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default='data/')
args = parser.parse_args()

app = Flask(__name__)
CORS(app)

system = BigBrother()
if not os.path.isdir(args.datadir):
    os.makedirs(args.datadir)
data_file = os.path.join(args.datadir, 'all_data.pkl')
if os.path.isfile(data_file):
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)
    system = BigBrother(all_data['embeddings'], all_data['names'],
                        all_data['top_k'], all_data['model'])


def image_from_list(list_image: List[List[int]]) -> np.ndarray:
    # expected that shape is height, width, channels (3)
    image = np.array(list_image, dtype=np.uint8)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f'image shape: {image.shape}')
    return image


@app.route('/get_nearest', methods=['GET'])
def get_nearest():
    data = request.get_json()  # Get data posted as a json
    # check that all the required parameters have provided
    expected_variables = ('image',)
    for variable in expected_variables:
        if variable not in data:
            answer = {'success': 0,
                      'message': f'Expected `{variable}` variable'}
            return answer

    # TODO: is it the most convenient format?
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
        distances, indices, names = system.get_nearest(**params)
        answer = {'success': 1, 'distances': distances, 'indices': indices,
                  'names': names}
    except Exception:
        answer = {'success': 0, 'message': 'Something went wrong'}

    return answer


@app.route('/add_face_photo', methods=['POST'])
def add_face_photo():
    data = request.get_json()  # Get data posted as a json
    # check that all the required parameters have provided
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

        # TODO: change writing to disk to writing to MongoDB
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
    # app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
