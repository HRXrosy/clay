tech load /work/probe2.tech -noprompt
puts "TECHNAME: [tech name]"
gds read /work/../../../cases/layout_intent/comparator_strongarm_tsmc40/final/comparator.gds
load comparator
select top cell
extract all
ext2spice lvs
ext2spice cthresh 0
ext2spice -o /work/probe2.spice
quit -noprompt
