# -*- coding:utf-8 -*-

import datetime
from common.mysql import *
from datasets.customer import *
from models.customer_loss import *

plat_form_db = get_db(host="192.168.2.240", user="yuanjun", password="123456")

customer = Customer(plat_form_db)

train_data = customer.query_customer_info()

customer_loss_model = CustomerLoss(2)
validation_data = None
start_time = datetime.datetime.now()
for x, y in train_data:
    if validation_data is None:
        validation_data = (x, y)
    else:
        customer_loss_model.fit(x, y, validation_data=validation_data, shuffle=True, batch_size=32, epochs=2)

        now = datetime.datetime.now()
        if (now - start_time).seconds > 600:
            model_save_path = now.strftime("%y%m%d%H")+"customer_loss_train"
            customer_loss_model.save_model(model_save_path)

