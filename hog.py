import cv2
import numpy as np
import math

class Hog_descriptor():

    def __init__(self, img, cell_size=8, bin_size=9):
        self.img = img
        self.img = np.sqrt(img * 1.0 / float(np.max(img)))  # gamma=0.5 gamma校正法归一化 调节对比度 降低局部阴影和光照的影响
        self.img = self.img * 255   # 反归一化
        self.cell_size = cell_size  # 单元边长
        self.bin_size = bin_size    # 直方图块数
        self.angle_unit = 180 / self.bin_size  # 直方图每个块的大小

    # 主功能，计算图像的HOG描述符，顺便求 HOG - image特征图
    def extract(self):
        height, width = self.img.shape
        gradient_value, gradient_angle = self.global_gradient()  #计算图像每一个像素点的梯度和方向
        gradient_value = abs(gradient_value)
        cell_gradient_value = np.zeros((height//self.cell_size, width//self.cell_size, self.bin_size))  #计算每个cell单元的梯度直方图，形成descriptor
        for i in range(cell_gradient_value.shape[0]):
            for j in range(cell_gradient_value.shape[1]):
                # 当前cell
                cs = self.cell_size
                cell_gradient = gradient_value[i*cs:i*cs+cs, j*cs:j*cs+cs]
                cell_angle = gradient_angle[i*cs:i*cs+cs, j*cs:j*cs+cs]
                cell_gradient_value[i][j] = self.cell_gradient(cell_gradient,cell_angle)    # 计算出该cell的直方图
        hog_vector = []     # 步长为一个cell大小，一个block由2x2个cell组成，遍历每一个block，串联得到所有block的HOG特征为最终特征向量
        for i in range(cell_gradient_value.shape[0] - 1):
            for j in range(cell_gradient_value.shape[1] - 1):
                block_vector = []   # 第[i][j]个block
                block_vector.extend(cell_gradient_value[i][j])
                block_vector.extend(cell_gradient_value[i][j + 1])
                block_vector.extend(cell_gradient_value[i + 1][j])
                block_vector.extend(cell_gradient_value[i + 1][j + 1])
                # 使用l2范数归一化
                divider = math.sqrt( sum(i**2 for i in block_vector) + 1e-5 )
                block_vector = [ i/divider for i in block_vector ]
                hog_vector.extend(block_vector)
        # 将得到的每个cell的梯度方向直方图绘出，得到特征图，便于观察
        hog_image = self.gradient_image(cell_gradient_value)
        return hog_vector, hog_image

    #  使用sobel算子计算每个像素沿x、y的梯度和方向
    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        gradient_magnitude, gradient_angle = cv2.cartToPolar(gradient_values_x, gradient_values_y,
                                                             angleInDegrees=True)
        gradient_angle[gradient_angle >= 180] -= 180
        gradient_angle[gradient_angle == 180] = 0
        return gradient_magnitude, gradient_angle

    # cell单元构建梯度方向直方图
    def cell_gradient(self, cell_gradient, cell_angle):
        bin = np.zeros(self.bin_size)
        # 遍历cell中的像素点
        for i in range(cell_gradient.shape[0]):
            for j in range(cell_gradient.shape[1]):
                value = cell_gradient[i][j]    # 当前像素的梯度幅值
                angle = cell_angle[i][j]       # 当前像素的梯度方向
                left_i = int(angle/self.angle_unit)%self.bin_size
                right_w = ( angle - left_i*self.angle_unit )/self.angle_unit

                bin[left_i] += value*(1-right_w)
                bin[(left_i+1)%self.bin_size] += value*right_w
        return bin

    # 将梯度直方图转化为特征图像,便于观察
    def gradient_image(self, cell_gradient):
        image=np.zeros(self.img.shape)
        cell_width = self.cell_size / 2
        max_mag = cell_gradient.max()
        # 遍历每一个cell
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y] / max_mag   # 获取第[i][j]个cell的梯度直方图 and 归一化
                # 遍历每一个bin区间
                for i in range(self.bin_size):
                    value = cell_grad[i]
                    angle_radian = math.radians(i*20)
                    # 计算起始坐标和终点坐标，长度为幅值(归一化),幅值越大、绘制的线条越长、越亮
                    x1 = int(x * self.cell_size + cell_width + value * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + cell_width + value * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size + cell_width - value * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size + cell_width - value * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(value)))
        return image

if __name__ == '__main__':
    ori_img= cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图
    hog_vec,hog_img = Hog_descriptor(ori_img).extract()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.4, 2.0 * 3.2))
    plt.subplot(1, 2, 1)
    plt.imshow(ori_img, cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.imshow(hog_img, cmap=plt.cm.gray)  # 输出灰度图
    plt.show()