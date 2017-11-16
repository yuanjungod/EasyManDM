# -*- coding:utf-8 -*-

import datetime
from common.mysql import *
from datasets.customer import *
from models.customer_loss import *

plat_form_db = get_db(host="192.168.2.240", user="yuanjun", password="123456")

customer_loss_model = CustomerLoss(2)
validation_data = None
start_time = datetime.datetime.now()
initial_epoch = 0
for train_date in Customer(plat_form_db).query_customer_info():
    x = train_date[0]
    y = train_date[1]
    if validation_data is None:
        print(sum(y) * 1.0 / len(y))
        validation_data = (x, y)
    else:
        print(sum(y) * 1.0 / len(y))
        epochs = 30
        print("initial_epoch %s" % initial_epoch)
        customer_loss_model.fit(
            x=x, y=y, shuffle=True, batch_size=200, epochs=epochs,
            initial_epoch=initial_epoch)
        print(customer_loss_model.get_model_para())
        initial_epoch += epochs
        print("train acc %s" % str(float(np.sum(np.argmax(customer_loss_model.predict_on_batch(x), axis=1)
                                        == np.argmax(y, axis=1))) / len(y)))
        print("test acc %s" % str(float(np.sum(np.argmax(customer_loss_model.predict_on_batch(validation_data[0]), axis=1)
                                       == np.argmax(validation_data[1], axis=1))) / len(validation_data[0])))
        # print customer_loss_model.predict_on_batch(validation_data[0])

        now = datetime.datetime.now()
        print((now - start_time).seconds % 90 == 0)
        if (now - start_time).seconds % 90 == 0:
            model_save_path = now.strftime("%y%m%d%H") + "customer_loss_train"
            customer_loss_model.save_model(model_save_path)
