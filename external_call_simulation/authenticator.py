import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from scipy import misc
import os
from src.verification_server import VerificationServer

def main():
    print('create verification server')
    verification_server = VerificationServer()
    print('verification server created')

    image_path = '/media/jonas/data/git_repositories/facenet/data/images/Anthony_Hopkins_0001.jpg'
    image = misc.imread(os.path.expanduser(image_path), mode='RGB')
    score = verification_server.compute_score(image)

    if score:
        print('Authenticator got the score ' + str(score))
    else:
        print('Authenticatee failed the required security checks and should be disabled')

if __name__ == '__main__':
    main()
