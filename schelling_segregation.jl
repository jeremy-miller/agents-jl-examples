using Agents
using GLMakie
using Random
using Statistics: mean

@agent SchellingAgent GridAgent{2} begin
    mood::Bool
    group::Int
end

function agent_step!(agent, model)
    minhappy = model.min_to_be_happy
    count_neighbors_same_group = 0
    for neighbor in nearby_agents(agent, model)
        if agent.group == neighbor.group
            count_neighbors_same_group += 1
        end
    end
    if count_neighbors_same_group ≥ minhappy
        agent.mood = true
    else
        agent.mood = false
        move_agent_single!(agent, model)
    end
    return
end

function initialize(; total_agents=320, griddims=(20, 20), min_to_be_happy=3, seed=125)
    space = GridSpaceSingle(griddims, periodic=false)
    properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Random.Xoshiro(seed)
    model = UnremovableABM(
        SchellingAgent, space;
        properties, rng, scheduler=Schedulers.Randomly()
    )
    for n in 1:total_agents
        agent = SchellingAgent(n, (1, 1), false, n < total_agents / 2 ? 1 : 2)
        add_agent_single!(agent, model)
    end
    return model
end

model = initialize(; total_agents=300)

groupcolor(a) = a.group == 1 ? :blue : :orange
groupmarker(a) = a.group == 1 ? :circle : :rect
x(agent) = agent.pos[1]
adata = [(:mood, sum), (x, mean)]
alabels = ["happy", "avg. x"]
parange = Dict(:min_to_be_happy => 0:8)

figure, abmobs = abmexploration(
    model;
    agent_step!, dummystep, parange,
    ac=groupcolor, am=groupmarker, as=10,
    adata, alabels
)

figure
