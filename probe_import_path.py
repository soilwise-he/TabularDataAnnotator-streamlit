import zenodo_autopilot
print(zenodo_autopilot.__file__)
print('has_patch', 'matcher_module.__dict__["model"] = model' in open(zenodo_autopilot.__file__, encoding='utf-8').read())
