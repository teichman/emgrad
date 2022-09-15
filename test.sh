
# Just make sure they all finish in a reasonable amount of time.
for BBF in $(seq 0 4); do
    echo "============================================================"
    echo "= Black box function $BBF (emgrad)"
    echo "============================================================"
    python3 emgrad.py --bbf $BBF
done

# Compare with autograd 
for BBF in $(seq 0 4); do
    echo "============================================================"
    echo "= Black box function $BBF (autograd comparison)"
    echo "============================================================"
    python3 emgrad.py --autograd --bbf $BBF
done
