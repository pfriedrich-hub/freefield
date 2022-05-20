import freefield
from freefield import DIR

proc_list = [['RX81', 'RX8', DIR / 'data' / 'play_buf_pulse.rcx'],
             ['RX82', 'RX8', DIR / 'data' / 'play_buf_pulse.rcx']]
freefield.initialize('dome', device=proc_list)

def single_chan_test(proc, chan):  # play from single channel
    print('play from ' + proc + ' channel ' + str(chan))
    freefield.write(tag='chan', value=chan, procs=proc)
    freefield.write(tag='playbuf', value=wnoise, procs=proc)
    freefield.play()

def chan_iter_test():     # iterate and play test sound over all channels
    wnoise = slab.Sound.whitenoise(duration=1)
    chan_list = list(range(0, 24))
    for proc in proc_list:
        proc = proc[0]
        for chan in chan_list:
            print(proc + ' ' + str(chan))
            input = 'proceed? 0 / 1'
            if input:
                print('play from ' + proc + ' channel ' + str(chan))
                freefield.write(tag='chan', value=channel, procs=proc)
                freefield.write(tag='playbuf', value=wnoise, procs=proc)
                freefield.play()