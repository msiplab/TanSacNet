import os
import requests
from torch.nn.functional import pad

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

def fcn_extract_blks(img, iBlk, blksz, k):
    """ 
    Function to extract a local patch block from a global array
    """
    # Extend array x
    kx = k[0]
    ky = k[1]
    iBlkRow = iBlk[0]
    iBlkCol = iBlk[1]
    padsz = ((kx-1)//2*blksz[1], (kx-1)//2*blksz[1], (ky-1)//2*blksz[0], (ky-1)//2*blksz[0])
    x = img.unsqueeze(0)
    xx = pad(x, padsz, mode='circular')
    posy = iBlkRow*blksz[0]
    posx = iBlkCol*blksz[1]
    y = xx[:, :, posy:posy+ky*blksz[0], posx:posx+kx*blksz[1]]
    return y.squeeze(0)

def fcn_place_blks(y,blk,iBlk,blksz):
    """ 
    Function to place a local patch block to a global array
    """
    # Place array blk 
    iBlkRow = iBlk[0]
    iBlkCol = iBlk[1]
    posy = iBlkRow*blksz[0]
    posx = iBlkCol*blksz[1]
    y[posy:posy+blksz[0], posx:posx+blksz[1]] = blk
    return y
