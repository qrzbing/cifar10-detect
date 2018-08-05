# coding:utf-8
import sys
from PyQt5.QtWidgets \
    import QApplication, QWidget, QPushButton, \
    QHBoxLayout, QVBoxLayout, QLabel, \
    QMainWindow, qApp, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5 import QtCore
import red_test
import cv2
import time

MAX_SHOW_HEIGHT = 650
MAX_SHOW_WIDTH = 800


class Cifar10_app(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()  # 界面绘制交给InitUi方法

    def resize_input_pic(self):
        wd = self.pixmap.width()
        hg = self.pixmap.height()
        if wd == 0 or hg == 0:
            return False
        if wd > hg:
            self.pixmap = self.pixmap.scaled(
                MAX_SHOW_WIDTH,
                MAX_SHOW_WIDTH/wd * hg
            )
        else:
            self.pixmap = self.pixmap.scaled(
                MAX_SHOW_HEIGHT/hg*wd,
                MAX_SHOW_HEIGHT
            )
        return True

    def initUI(self):
        ft = QFont()
        ft.setPointSize(25)
        # 初始化神经网络
        self.pre_test = red_test.ConvNet(
            n_channel=3, n_classes=10,
            image_size=24, n_layers=20
        )
        # 设置窗口的图标，引用当前目录下的2.jpg图片
        self.setWindowIcon(QIcon('ico.ico'))

        # 两个button
        btn_file = QPushButton(u'导入文件')
        btn_file.setFont(ft)
        btn_file.setToolTip(u'从这里选择文件并导入')
        btn_file.clicked.connect(self.btn_file_clicked)

        btn_det = QPushButton(u'开始检测')
        btn_det.setFont(ft)
        btn_det.setToolTip(u'你懂得')
        btn_det.clicked.connect(self.btn_det_clicked)
        btn_exit = QPushButton(u'退出')
        btn_exit.setFont(ft)
        btn_exit.setToolTip(u'退出')
        btn_exit.clicked.connect(qApp.quit)
        # 加载图片
        self.pixmap = QPixmap('team2.jpg')
        resize_flag = self.resize_input_pic()
        if resize_flag == False:
            assert(u'加载图片失败')

        self.lbl_pic = QLabel(self)
        self.lbl_pic.setPixmap(self.pixmap)

        self.lbl_sta = QLabel(self)
        self.lbl_sta.setFont(ft)
        self.lbl_sta.setText('Written By bilibili - qrz')
        self.lbl_sta.resize(500, 100)
        self.pic_name = ""

        hbox = QHBoxLayout()
        hbox.addWidget(btn_file)
        hbox.addWidget(btn_det)
        hbox.addWidget(btn_exit)

        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl_pic)
        vbox.addWidget(self.lbl_sta)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        # 设置窗口的位置和大小
        self.setGeometry(50, 50, 800, 700)

        # 设置窗口的标题
        self.setWindowTitle(u'分类识别应用')
        # 显示窗口
        self.show()

    def btn_file_clicked(self):
        # sender = self.sender()
        # self.lbl_sta.setText('push button clicked')
        # exit(0)
        # qApp.quit
        self.pic_name, _ = QFileDialog.getOpenFileName(
            self,
            u'选取文件',
            './images',
            'All Files (*);;JPEG Files(*.jpg);;PNG Files(*.png)'
        )
        # pass
        self.pixmap = QPixmap(self.pic_name)
        input_file_flag = self.resize_input_pic()
        if self.pic_name == "":
            self.lbl_sta.setText(u'文件导入失败或未导入文件!')
            return
        if not self.pic_name[-3:] in ("jpg", "png", "bmp"):
            self.lbl_sta.setText(u"该文件不是图片!")
            return
        self.lbl_pic.setPixmap(self.pixmap)
        self.lbl_sta.setText(u'[+] 成功加载文件: ' + self.pic_name)
        # print(pic_name)

    def btn_det_clicked(self):
        if self.pic_name == '':
            self.lbl_sta.setText(u"请在导入文件后再开始检测!")
            return
        self.lbl_sta.setText(u"正在识别中......")
        # 第一次检测
        if sys.platform == "win32":
            self.start_time = time.clock()
        else:
            self.start_time = time.time()
        pre_list1 = self.pre_test.test(
            backup_path='backup/cifar10-v16/',
            epoch=500, image_str=self.pic_name
        )
        if sys.platform == "win32":
            self.end_time = time.clock()
        else:
            self.end_time = time.time()
        used_time1 = self.end_time - self.start_time
        # 第二次检测
        if sys.platform == "win32":
            self.start_time = time.clock()
        else:
            self.start_time = time.time()
        pre_list2 = self.pre_test.test(
            backup_path='backup/cifar10-v16/',
            epoch=500, image_str=self.pic_name
        )
        if sys.platform == "win32":
            self.end_time = time.clock()
        else:
            self.end_time = time.time()
        used_time2 = self.end_time - self.start_time
        show_text = [
            u'这是飞机(airplane)',
            u'这是汽车(automobile)',
            u'这是只鸟(bird)',
            u'这是只猫(cat)',
            u'这是只鹿(deer)',
            u'这是条狗(dog)',
            u'这是青蛙(frog)',
            u'这是匹马(horse)',
            u'这是条船(boat)',
            u'这是卡车(truck)'
        ]
        flag = False
        txt_0 = u'[+] 文件: ' + self.pic_name + '\n'
        for i in range(10):
            if pre_list1[i] == 1.0:
                txt_1 = u"[+] 第一次检测感觉" + show_text[i] + \
                    u"\n一共用时" + str(used_time1) + u"秒\n"
                flag = True
                break
        if flag is False:
            txt_1 = u'呜...识别不出来'

        for i in range(10):
            if pre_list2[i] == 1.0:
                txt_2 = u"[+] 第二次检测感觉" + show_text[i] + \
                    u"\n一共用时" + str(used_time2) + u"秒"
                flag = True
                break
        if flag is False:
            txt_2 = u'呜...识别不出来'
        txt_all = txt_0 + txt_1 + txt_2
        self.lbl_sta.setText(txt_all)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Cifar10_app()
    sys.exit(app.exec_())
