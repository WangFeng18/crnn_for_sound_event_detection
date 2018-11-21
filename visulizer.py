import os
import sed_vis
import dcase_util
import parameter
taudio = 'sed_vis/tests/data/a001.wav'
tgt = 'sed_vis/tests/data/a001.ann'
tout = 'sed_vis/tests/data/a001_system_output.ann'
id = '16'
audio_path = os.path.join(parameter.audio_dir, parameter.audio_template.format(id))
gt_path = os.path.join(parameter.meta_dir, parameter.meta_template.format(id))
output_path = os.path.join(parameter.estimate_dir, parameter.meta_template.format(id))
def vis(audio_path, gt_path, output_path=None):
    audio_container = dcase_util.containers.AudioContainer().load(audio_path)
    reference_event_list = dcase_util.containers.MetaDataContainer().load(gt_path)
    event_lists = {'reference': reference_event_list}
    if output_path is not None:
        estimated_event_list = dcase_util.containers.MetaDataContainer().load(output_path)
        event_lists.update({'estimated': estimated_event_list})
    vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists, audio_signal=audio_container.data, sampling_rate=audio_container.fs, spec_hop_length=320, event_roll_cmap='gnuplot', publication_mode=True)
    vis.show()

vis(audio_path, gt_path, output_path)
