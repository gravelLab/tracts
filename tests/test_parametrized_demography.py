from tracts import ParametrizedDemography


def test():
    model = ParametrizedDemography()
    model.add_founder_event({'A': 'm1_A'}, 'B', 't0')
    model.add_pulse_migration('C', 'r1', 't1')
    model.add_pulse_migration('D', 'r1', 't2')
    model.finalize()
    m = model.get_migration_matrix([0.1, 4, 0.2, 1, 2.2])
    print(m)
    print(model.free_params)


# def test_2():
#     model = ParametrizedDemography.load_from_YAML('pp_px.yaml')
#     m = model.get_migration_matrix([0.2, 4, 0.375, 3])
#     print(m)
#     print(model.proportions_from_matrix(m))
#     print(model.free_params)
#     model.fix_ancestry_proportions('r', [0.5, 0.5])
#     print(model.params_fixed_by_ancestry)
#     model.get_migration_matrix([0.2, 4, 0.375])


def test_3():
    model = ParametrizedDemography.load_from_YAML('pp.yaml')
    m = model.get_migration_matrix([0.2, 4.1])
    print(m)