COMMENT
Since this is an electrode current, positive values of i depolarize the cell
and in the presence of the extracellular mechanism there will be a change
in vext since i is not a transmembrane current but a current injected
directly to the inside of the cell.
ENDCOMMENT

NEURON {
        POINT_PROCESS SinSyn
        RANGE del, dur, pkamp, freq, phase, bias
        NONSPECIFIC_CURRENT i
}

UNITS {
        (nA) = (nanoamp)
             }

PARAMETER {
        del=5   (ms)
        dur=200   (ms)
        pkamp=1 (nA)
        freq=1  (Hz)
        phase=0
        bias=0  (nA)
        PI=3.14159265358979323846
}

ASSIGNED {
        i (nA)
}

BREAKPOINT {
       at_time(del)
       at_time(del + dur)

       if (t < del) {
      i=0   
   }else{ 
            if (t < del+dur) {
           i = -pkamp*sin(2*PI*freq*(t-del)*(0.001)+phase)-bias
      }else{ 
           i = 0
}}}

NET_RECEIVE(weight (nA)) {
}
