import numpy as np 
import tensorflow as tf 
from time import time
import math

from include.data import get_data_set
from include.model import model, lr_decay

# dataset 
X_train, y_train, X_test, y_test = get_data_set()
tf.set_random_seed(21)
# training examples
n_obs = X_train.shape[0]


X, Y, output, y_pred_cls, global_step, learning_rate = model("model_ujh_f000")
global_accuracy = 0
epoch_start = 0

# parameters
_BATCH_SIZE = 512
_EPOCH = 5
save_path = './graph'

# loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(loss, global_step=global_step)

# predict and accuracy
correct_prediction = tf.equal(y_pred_cls, tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# saver
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(save_path, sess.graph)


try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def train(epoch):
    global epoch_start
    
    epoch_start = time()
    
    batch_size = int(math.ceil((len(X_train) /_BATCH_SIZE)))
    i_global = 0
    
    for s in range(batch_size):
        batch_xs = X_train[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = y_train[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        
        start_time = time()
        # sess.run(tf.global_variables_initializer())
        i_global, _, batch_loss, batch_acc = sess.run(
                        [global_step, optimizer, loss, accuracy],
                        feed_dict={X:batch_xs, Y:batch_ys, learning_rate:lr_decay(epoch, n_obs, _BATCH_SIZE)})
        # duration time 持续时间
        duration = time() - start_time
        
        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))
            
            # 显示进度条
            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))
        
    
    test_and_save(i_global, epoch)
    

def test_and_save(_global_step, epoch):
    
    global global_accuracy
    global epoch_start
    
    i = 0
    # predict
    predicted_class = np.zeros(shape=len(X_test), dtype=np.int)
    
    while i < len(X_test):
        j = min(i + _BATCH_SIZE, len(X_test))
        batch_xs = X_test[i:j, :]
        batch_ys = y_test[i:j, :]
        
        predicted_class[i:j] = sess.run(
                    y_pred_cls,
                    feed_dict={X:batch_xs, Y:batch_ys, learning_rate:lr_decay(epoch, n_obs, _BATCH_SIZE)})
        i = j
    
    correct = (np.argmax(y_test, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
    
    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{}) - time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format((epoch+1), acc, correct_numbers, len(X_test), int(hours), int(minutes), seconds))
    
    if global_accuracy != 0 and global_accuracy < acc:
        
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Accuracy/test', simple_value=acc),
        ])
        train_writer.add_summary(summary, _global_step)
        
        saver.save(sess, save_path=save_path, global_step=_global_step)
        
        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("###########################################################################################################")

    
def main():
    
    train_start = time()

    for i in range(_EPOCH):
        print("\nEpoch: {}/{}\n".format((i+1), _EPOCH))
        train(i)

    hours, rem = divmod(time() - train_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "Best accuracy pre session: {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format(global_accuracy, int(hours), int(minutes), seconds))
    

if __name__ == "__main__":
    main()

sess.close()
