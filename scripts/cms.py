from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as metrics

if __name__ == '__main__':

    # crema_d_average_cm = np.array([[178.40,	18.60,	3.00,	19.80,	26.20,	8.20],
    #                                [34.40,	169.40,	6.00,	16.60,	10.40,	17.40],
    #                                [3.60,	2.80,	235.00,	4.20,	4.20,	4.40],
    #                                [25.80,	6.60,	2.20,	208.40,	2.00,	9.20],
    #                                [27.00,	3.80,	3.00,	1.80,	172.00,	9.80],
    #                                [10.80,	8.40,	4.80,	8.20,	11.80,	210.20]])
    # crema_d_average_cm = np.floor(crema_d_average_cm)
    # crema_d_average_cm = crema_d_average_cm.astype(int)

    # display = metrics.ConfusionMatrixDisplay(crema_d_average_cm, display_labels=[
    #                                          'Sad',	'Fear',	'Happy',	'Disgust',	'Neutral',	'Anger'])
    # display.plot()
    # plt.savefig('/your/folder/crema_d_average_cm.png')

    # ravdess_average_cm = np.array([[26.80,	3.20, 0.00, 3.20, 6.00, 0.00, 0.00, 0.80],
    #                                [3.80,	66.00,	0.80,	4.80,
    #                                    4.60,	0.00,	0.00, 0.00],
    #                                [0.00,	2.20, 72.40,	2.00,
    #                                    0.00,	0.00,	0.80, 2.60],
    #                                [4.40,	6.60, 2.00, 46.80,
    #                                    6.00,	6.40,	5.60, 2.20],
    #                                [1.80,	1.20, 0.40, 4.00,
    #                                    65.20,	2.20,	2.00, 3.20],
    #                                [2.00,	1.60, 1.60, 4.60,
    #                                    5.60, 54.60,	4.40, 5.60],
    #                                [0.00,	0.40, 1.20, 6.80,
    #                                    6.40, 0.80, 63.80,	0.60],
    #                                [0.00,	1.20, 5.20, 2.60, 15.60,	12.4,	1.00, 42.00]])
    # ravdess_average_cm = np.floor(ravdess_average_cm)
    # ravdess_average_cm = ravdess_average_cm.astype(int)

    # display = metrics.ConfusionMatrixDisplay(ravdess_average_cm, display_labels=[
    #                                          'Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprise'])
    # display.plot()
    # plt.savefig('/your/folder/ravdess_average_cm.png')

    ravdess_to_crema_d = np.array([[258.40,	118.20,	11.60,	294.40,	151.40,	26.40,	325.60,	85.00],
                                   [92.40, 420.00,	57.80,	231.60,
                                       51.40,	18.00,	106.80,	293.00],
                                   [9.20, 17.00,	855.00,	27.40,
                                       11.40,	8.00,	125.40,	217.60],
                                   [44.80, 32.40,	49.20, 935.60,
                                       5.80, 17.00,	91.80, 94.40],
                                   [103.40,	90.00,	5.40,	109.80,
                                       365.60,	42.40,	316.80,	53.60],
                                   [109.40,	124.80, 47.20,	194.80,
                                       66.00, 390.00,	103.00,	235.80],
                                   [0.00, 0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00],
                                   [0.00, 0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00]])
    ravdess_to_crema_d = np.floor(ravdess_to_crema_d)
    ravdess_to_crema_d = ravdess_to_crema_d.astype(int)

    display = metrics.ConfusionMatrixDisplay(ravdess_to_crema_d, display_labels=[
                                             'Sad',	'Fear',	'Happy', 'Disgust',	'Neutral',	'Anger', 'Calm', 'Surprise'])
    display.plot()
    plt.savefig('/your/folder/ravdess_to_crema_average_cm.png')

    crema_d_to_ravdess = np.array([[174.00,	0.00,	0.00,	15.80,	2.20,	0.00,	0.00,	0.00],
                                   [173.60,	0.00,	84.00,	108.60,
                                       0.80,	0.00,	17.00,	0.00],
                                   [8.60,	0.00,	355.80,	0.80,
                                       0.00, 9.40,	9.40,	0.00],
                                   [68.60,	0.00,	12.40,	203.80,
                                       7.80, 9.80,	81.60,	0.00],
                                   [26.20,	0.00,	1.40,	13.20,
                                       284.20,	9.80,	49.20,	0.00],
                                   [33.00,	0.00,	11.60,	42.00,
                                       16.40,	215.20,	65.80,	0.00],
                                   [3.60, 0.00,	0.00,	13.60,
                                       0.20,	3.00, 363.60,	0.00],
                                   [29.20, 0.00,	55.40,	29.00,	56.80,	160.40,	53.20,	0.00]])
    crema_d_to_ravdess = np.floor(crema_d_to_ravdess)
    crema_d_to_ravdess = crema_d_to_ravdess.astype(int)

    display = metrics.ConfusionMatrixDisplay(crema_d_to_ravdess, display_labels=[
                                             'Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprise'])
    display.plot()
    plt.savefig('/your/folder/crema_d_to_ravdess_average_cm.png')
