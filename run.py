import sys
import time

import cv2
from torch.autograd import Variable
from PyQt5.QtWidgets import QWidget, QApplication, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from network import *


use_cuda = 0
style_names = ('autoportrait', 'candy', 'composition', 'edtaonisl', 'udnie')
style_model_path = 'models/weights/'
style_img_path = 'models/style/'


class ReCoNetTransfer:
    def __init__(self):
        self.model = ReCoNet()
        self.model_path = None

    def transfer(self, x_np):
        if self.model_path is None:
            return x_np, None

        x_np = x_np.transpose(2, 0, 1)
        x_tensor = torch.from_numpy(x_np).float().unsqueeze(0)
        x_variable = Variable(x_tensor.cuda() if use_cuda else x_tensor, volatile=True)

        inference_time = time.time()
        y_variable = self.model(x_variable)
        inference_time = time.time() - inference_time

        y_np = y_variable.cpu().clamp(0, 255).data.squeeze(0).numpy().transpose(1, 2, 0).astype('uint8').copy()

        return y_np, inference_time

    def change_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        if use_cuda:
            self.model.cuda()
        self.model.eval()
        self.model_path = model_path
        return


class Thread(QThread):
    change_pixmap_x = pyqtSignal(QPixmap)
    change_pixmap_y = pyqtSignal(QPixmap)
    change_pixmap_inf = pyqtSignal(str)
    change_pixmap_dis = pyqtSignal(str)

    def __init__(self, video_width, video_height, parent=None):
        QThread.__init__(self, parent=parent)
        self.video_height = video_height
        self.video_width = video_width
        self.reconet = ReCoNetTransfer()

    def transfer(self, x_np):
        return self.reconet.transfer(x_np)

    def change_model(self, model_path):
        self.reconet.change_model(model_path)
        return

    def run(self):
        cap = cv2.VideoCapture(0)
        fps_update_cnt = 0
        fps_update_num = 10
        while True:
            display_time = time.time()
            ret, x_np = cap.read()
            x_np = cv2.cvtColor(x_np, cv2.COLOR_BGR2RGB)

            y_np, inference_time = self.transfer(x_np)

            x_qt = QImage(x_np.data, x_np.shape[1], x_np.shape[0], QImage.Format_RGB888)
            x_qt = QPixmap.fromImage(x_qt)
            x_qt = x_qt.scaled(self.video_width, self.video_height, Qt.KeepAspectRatio)

            y_qt = QImage(y_np.data, y_np.shape[1], y_np.shape[0], QImage.Format_RGB888)
            y_qt = QPixmap.fromImage(y_qt)
            y_qt = y_qt.scaled(self.video_width, self.video_height, Qt.KeepAspectRatio)

            self.change_pixmap_x.emit(x_qt)
            self.change_pixmap_y.emit(y_qt)

            fps_update_cnt = (fps_update_cnt + 1) % fps_update_num
            if fps_update_cnt == 0:
                self.change_pixmap_inf.emit('    Infrence FPS: {0:.2f}'.format(
                    1 / inference_time if inference_time is not None else 0))
                display_time = time.time() - display_time
                self.change_pixmap_dis.emit('    Display FPS: {0:.2f}'.format(1 / display_time))


class StyleLabel(QLabel):
    signal = pyqtSignal(['QString'])

    def __init__(self, label, parent=None):
        QLabel.__init__(self, parent=parent)
        self.model_name = style_model_path + style_names[label] + '.model'

    def mousePressEvent(self, event):
        self.signal.emit(self.model_name)


class GUI(QWidget):
    def __init__(self, video_width=640, video_height=480, padding=20, margin=100):
        super().__init__()
        self.title = 'ReCoNet Demo'

        self.video_width, self.video_height, self.padding, self.margin = video_width, video_height, padding, margin
        self.width = self.video_width * 2 + self.padding * 3

        style_num = len(style_names)
        self.style_size = (self.width - self.padding * (style_num + 1)) // style_num
        self.height = self.video_height + self.style_size + self.padding * 3
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.margin, self.margin, self.width, self.height)

        label_x = QLabel(self)
        label_x.move(self.padding, self.padding)
        label_x.resize(self.video_width, self.video_height)
        label_y = QLabel(self)
        label_y.move(self.video_width + self.padding * 2, self.padding)
        label_y.resize(self.video_width, self.video_height)

        label_inf = QLabel(self)
        label_inf.move(self.video_width + self.padding * 2, self.padding)
        label_inf.resize(150, 50)
        label_inf.setStyleSheet('background: yellow')
        label_dis = QLabel(self)
        label_dis.move(self.padding, self.padding)
        label_dis.resize(150, 50)
        label_dis.setStyleSheet('background: yellow')

        th = Thread(self.video_width, self.video_height, parent=self)
        th.change_pixmap_x.connect(label_x.setPixmap)
        th.change_pixmap_y.connect(label_y.setPixmap)
        th.change_pixmap_inf.connect(label_inf.setText)
        th.change_pixmap_dis.connect(label_dis.setText)

        label_style = []
        for i in range(len(style_names)):
            tmp_label = StyleLabel(i, self)
            tmp_label.move((self.style_size + self.padding) * i + self.padding, self.video_height + self.padding * 2)
            tmp_label.resize(self.style_size, self.style_size)
            tmp_pixmap = QPixmap('models/style/' + style_names[i] + '.jpg').scaled(self.style_size, self.style_size)
            tmp_label.setPixmap(tmp_pixmap)
            tmp_label.signal.connect(th.change_model)
            label_style.append(tmp_label)
        th.start()

        self.show()


def main():
    app = QApplication(sys.argv)
    gui = GUI()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
