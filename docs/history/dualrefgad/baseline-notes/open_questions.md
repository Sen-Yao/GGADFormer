# VecGAD Open Questions For DualRefGAD

1. Is VecGAD's Photo advantage mainly from high-dimensional raw attributes, or
   from graph tokenization?
2. Does a simple known-normal attribute residual reproduce a large fraction of
   VecGAD's Photo score?
3. Are DualRefGAD's Photo false negatives nodes with high normal-explanation
   failure but weak/reversed `mat_mean`?
4. Does response `range` proxy the same direction as VecGAD's discrepancy
   vector, or is it a separate signal?
5. Can a label-free regime variable predict when DualRefGAD should use
   mean-like, prefix-like, range-like, or unordered set readout?
6. Would generated hard negatives help DualRefGAD, or would they break the
   current reference-evidence story?
7. Are reported VecGAD and DualRefGAD Photo splits exactly comparable?
