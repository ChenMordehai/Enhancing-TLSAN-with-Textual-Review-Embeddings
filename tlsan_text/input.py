import numpy as np

# training
class DataInput:
    def __init__(self, data, batch_size, k):
        self.k = k
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        self.i = 0  # reset for each new iteration
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1

        # Build lists from dictionary keys
        u, target_item, y, sl, new_sl, c, r = [], [], [], [], [], [], []
        for t in ts:
            u.append(t['reviewerID'])
            target_item.append(t['target_item'])
            y.append(t['label'])
            c.append(t['current_category'])
            sl.append(min(len(t['history']), self.k))
            new_sl.append(len(t['session']))
            r.append(t['review_embedding'])
        max_new_sl = max(new_sl)

        hist_i = np.zeros([len(ts), self.k], np.int64)
        hist_t = np.zeros([len(ts), self.k], np.float32)
        hist_i_new = np.zeros([len(ts), max_new_sl], np.int64)

        kk = 0
        for t in ts:
            length = len(t['history'])
            if length > self.k:
                for l in range(self.k):
                    hist_i[kk][l] = t['history'][length - self.k + l]
                    hist_t[kk][l] = t['time_embedding'][length - self.k + l]
            else:
                for l in range(length):
                    hist_i[kk][l] = t['history'][l]
                    hist_t[kk][l] = t['time_embedding'][l]
            for l in range(len(t['session'])):
                hist_i_new[kk][l] = t['session'][l]
            kk += 1

        # The returned tuple now has 10 elements:
        # (u, target_item, y, hist_i, hist_i_new, hist_t, sl, new_sl, current_category, review_embedding)
        return self.i, (u, target_item, y, hist_i, hist_i_new, hist_t, sl, new_sl, c, r)


# testing
class DataInputTest:
    def __init__(self, data, batch_size, k):
        self.k = k
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        self.i = 0  # reset for each new iteration
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1

        # For test, note that the target is a dict with positive/negative keys.
        u, pos_item, neg_item, sl, new_sl, c, r = [], [], [], [], [], [], []
        for t in ts:
            u.append(t['reviewerID'])
            pos_item.append(t['target_items']['positive'])
            neg_item.append(t['target_items']['negative'])
            c.append(t['current_category'])
            sl.append(min(len(t['history']), self.k))
            new_sl.append(len(t['session']))
            r.append(t['review_embedding'])
        max_new_sl = max(new_sl)

        hist_i = np.zeros([len(ts), self.k], np.int64)
        hist_t = np.zeros([len(ts), self.k], np.float32)
        hist_i_new = np.zeros([len(ts), max_new_sl], np.int64)

        kk = 0
        for t in ts:
            length = len(t['history'])
            if length > self.k:
                for l in range(self.k):
                    hist_i[kk][l] = t['history'][length - self.k + l]
                    hist_t[kk][l] = t['time_embedding'][length - self.k + l]
            else:
                for l in range(length):
                    hist_i[kk][l] = t['history'][l]
                    hist_t[kk][l] = t['time_embedding'][l]
            for l in range(len(t['session'])):
                hist_i_new[kk][l] = t['session'][l]
            kk += 1

        # Returned tuple: (u, pos_item, neg_item, hist_i, hist_i_new, hist_t, sl, new_sl, current_category, review_embedding)
        return self.i, (u, pos_item, neg_item, hist_i, hist_i_new, hist_t, sl, new_sl, c, r)
