import os
import requests
#import torchvision

def fcn_download_img(isVerbose=None):
    """
      FCN_DOWNLOAD_IMG
        Download Kodak Lossless True Color Image Suite.
        https://www.r0k.us/graphics/kodak/
    """

    # デフォルトの値を設定
    if isVerbose is None:
        isVerbose = True

    # 画像サンプルのダウンロード
    dstdir = '../../../data/'

    for idx in range(1, 25):  
        fname = "kodim{:02d}.png".format(idx)
        if not os.path.exists(os.path.join(dstdir, fname)):
            url = "https://www.r0k.us/graphics/kodak/kodak/" + fname
            response = requests.get(url)
            img = response.content
            with open(os.path.join(dstdir, fname), 'wb') as f:
                f.write(img)
            if isVerbose:
                print('Downloaded and saved {} in {}'.format(fname, dstdir))
        else:
            if isVerbose:
                print('{} already exists in {}'.format(fname, dstdir))        
    print('See Kodak Lossless True Color Image Suite (https://www.r0k.us/graphics/kodak/)')
    
    return dstdir