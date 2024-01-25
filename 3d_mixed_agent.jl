using Agents, Agents.Pathfinding
using Random
import ImageMagick
using FileIO: load
using GLMakie

@agent Animal ContinuousAgent{3} begin
    type::Symbol # one of :rabbit, :fox or :hawk
    energy::Float64
end

const v0 = (0.0, 0.0, 0.0) # we don't use the velocity field here
Rabbit(id, pos, energy) = Animal(id, pos, v0, :rabbit, energy)
Fox(id, pos, energy) = Animal(id, pos, v0, :fox, energy)
Hawk(id, pos, energy) = Animal(id, pos, v0, :hawk, energy)
eunorm(vec) = √sum(vec .^ 2)

function initialize_model(
    heightmap_url=
    "https://raw.githubusercontent.com/JuliaDynamics/" *
    "JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png",
    water_level=8,
    grass_level=20,
    mountain_level=35;
    n_rabbits=160,  ## initial number of rabbits
    n_foxes=30,  ## initial number of foxes
    n_hawks=30,  ## initial number of hawks
    Δe_grass=25,  ## energy gained from eating grass
    Δe_rabbit=30,  ## energy gained from eating one rabbit
    rabbit_repr=0.06,  ## probability for a rabbit to (asexually) reproduce at any step
    fox_repr=0.03,  ## probability for a fox to (asexually) reproduce at any step
    hawk_repr=0.02, ## probability for a hawk to (asexually) reproduce at any step
    rabbit_vision=6,  ## how far rabbits can see grass and spot predators
    fox_vision=10,  ## how far foxes can see rabbits to hunt
    hawk_vision=15,  ## how far hawks can see rabbits to hunt
    rabbit_speed=1.3, ## movement speed of rabbits
    fox_speed=1.1,  ## movement speed of foxes
    hawk_speed=1.2, ## movement speed of hawks
    regrowth_chance=0.03,  ## probability that a patch of grass regrows at any step
    dt=0.1,   ## discrete timestep each iteration of the model
    seed=42,  ## seed for random number generator
)

    # Download and load the heightmap. The grayscale value is converted to `Float64` and
    # scaled from 1 to 40
    heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
    # The x and y dimensions of the pathfinder are that of the heightmap
    dims = (size(heightmap)..., 50)
    # The region of the map that is accessible to each type of animal (land-based or flying)
    # is defined using `BitArrays`
    land_walkmap = BitArray(falses(dims...))
    air_walkmap = BitArray(falses(dims...))
    for i in 1:dims[1], j in 1:dims[2]
        # land animals can only walk on top of the terrain between water_level and grass_level
        if water_level < heightmap[i, j] < grass_level
            land_walkmap[i, j, heightmap[i, j]+1] = true
        end
        # air animals can fly at any height upto mountain_level
        if heightmap[i, j] < mountain_level
            air_walkmap[i, j, (heightmap[i, j]+1):mountain_level] .= true
        end
    end

    # Generate the RNG for the model
    rng = MersenneTwister(seed)

    # Note that the dimensions of the space do not have to correspond to the dimensions
    # of the pathfinder. Discretisation is handled by the pathfinding methods
    space = ContinuousSpace((100.0, 100.0, 50.0); periodic=false)

    # Generate an array of random numbers, and threshold it by the probability of grass growing
    # at that location. Although this causes grass to grow below `water_level`, it is
    # effectively ignored by `land_walkmap`
    grass = BitArray(
        rand(rng, dims[1:2]...) .< ((grass_level .- heightmap) ./ (grass_level - water_level)),
    )
    properties = (
        # The pathfinder for rabbits and foxes
        landfinder=AStar(space; walkmap=land_walkmap),
        # The pathfinder for hawks
        airfinder=AStar(space; walkmap=air_walkmap, cost_metric=MaxDistance{3}()),
        Δe_grass=Δe_grass,
        Δe_rabbit=Δe_rabbit,
        rabbit_repr=rabbit_repr,
        fox_repr=fox_repr,
        hawk_repr=hawk_repr,
        rabbit_vision=rabbit_vision,
        fox_vision=fox_vision,
        hawk_vision=hawk_vision,
        rabbit_speed=rabbit_speed,
        fox_speed=fox_speed,
        hawk_speed=hawk_speed,
        heightmap=heightmap,
        grass=grass,
        regrowth_chance=regrowth_chance,
        water_level=water_level,
        grass_level=grass_level,
        dt=dt,
    )

    model = ABM(Animal, space; rng, properties)

    # spawn each animal at a random walkable position according to its pathfinder
    for _ in 1:n_rabbits
        add_agent_pos!(
            Rabbit(
                nextid(model), ## Using `nextid` prevents us from having to manually keep track
                # of animal IDs
                random_walkable(model, model.landfinder),
                rand(model.rng, Δe_grass:2Δe_grass),
            ),
            model,
        )
    end
    for _ in 1:n_foxes
        add_agent_pos!(
            Fox(
                nextid(model),
                random_walkable(model, model.landfinder),
                rand(model.rng, Δe_rabbit:2Δe_rabbit),
            ),
            model,
        )
    end
    for _ in 1:n_hawks
        add_agent_pos!(
            Hawk(
                nextid(model),
                random_walkable(model, model.airfinder),
                rand(model.rng, Δe_rabbit:2Δe_rabbit),
            ),
            model,
        )
    end

    return model
end

model = initialize_model()

function animal_step!(animal, model)
    if animal.type == :rabbit
        rabbit_step!(animal, model)
    elseif animal.type == :fox
        fox_step!(animal, model)
    else
        hawk_step!(animal, model)
    end
end

function rabbit_step!(rabbit, model)
    # Eat grass at this position, if any
    if get_spatial_property(rabbit.pos, model.grass, model) == 1
        model.grass[get_spatial_index(rabbit.pos, model.grass, model)] = 0
        rabbit.energy += model.Δe_grass
    end

    # The energy cost at each step corresponds to the amount of time that has passed
    # since the last step
    rabbit.energy -= model.dt
    # All animals die if their energy reaches 0
    if rabbit.energy <= 0
        remove_agent!(rabbit, model, model.landfinder)
        return
    end

    # Get a list of positions of all nearby predators
    predators = [
        x.pos for x in nearby_agents(rabbit, model, model.rabbit_vision) if
        x.type == :fox || x.type == :hawk
    ]
    # If the rabbit sees a predator and isn't already moving somewhere
    if !isempty(predators) && is_stationary(rabbit, model.landfinder)
        # Try and get an ideal direction away from predators
        direction = (0.0, 0.0, 0.0)
        for predator in predators
            # Get the direction away from the predator
            away_direction = (rabbit.pos .- predator)
            # In case there is already a predator at our location, moving anywhere is
            # moving away from it, so it doesn't contribute to `direction`
            all(away_direction .≈ 0.0) && continue
            # Add this to the overall direction, scaling inversely with distance.
            # As a result, closer predators contribute more to the direction to move in
            direction = direction .+ away_direction ./ eunorm(away_direction)^2
        end
        # If the only predator is right on top of the rabbit
        if all(direction .≈ 0.0)
            # Move anywhere
            chosen_position = random_walkable(rabbit.pos, model, model.landfinder, model.rabbit_vision)
        else
            # Normalize the resultant direction, and get the ideal position to move it
            direction = direction ./ eunorm(direction)
            # Move to a random position in the general direction of away from predators
            position = rabbit.pos .+ direction .* (model.rabbit_vision / 2.0)
            chosen_position = random_walkable(position, model, model.landfinder, model.rabbit_vision / 2.0)
        end
        plan_route!(rabbit, chosen_position, model.landfinder)
    end

    # Reproduce with a random probability, scaling according to the time passed each
    # step
    rand(model.rng) <= model.rabbit_repr * model.dt && reproduce!(rabbit, model)

    # If the rabbit isn't already moving somewhere, move to a random spot
    if is_stationary(rabbit, model.landfinder)
        plan_route!(
            rabbit,
            random_walkable(rabbit.pos, model, model.landfinder, model.rabbit_vision),
            model.landfinder
        )
    end

    # Move along the route planned above
    move_along_route!(rabbit, model, model.landfinder, model.rabbit_speed, model.dt)
end

function fox_step!(fox, model)
    # Look for nearby rabbits that can be eaten
    food = [x for x in nearby_agents(fox, model) if x.type == :rabbit]
    if !isempty(food)
        remove_agent!(rand(model.rng, food), model, model.landfinder)
        fox.energy += model.Δe_rabbit
    end


    # The energy cost at each step corresponds to the amount of time that has passed
    # since the last step
    fox.energy -= model.dt
    # All animals die once their energy reaches 0
    if fox.energy <= 0
        remove_agent!(fox, model, model.landfinder)
        return
    end

    # Random chance to reproduce every step
    rand(model.rng) <= model.fox_repr * model.dt && reproduce!(fox, model)

    # If the fox isn't already moving somewhere
    if is_stationary(fox, model.landfinder)
        # Look for any nearby rabbits
        prey = [x for x in nearby_agents(fox, model, model.fox_vision) if x.type == :rabbit]
        if isempty(prey)
            # Move anywhere if no rabbits were found
            plan_route!(
                fox,
                random_walkable(fox.pos, model, model.landfinder, model.fox_vision),
                model.landfinder,
            )
        else
            # Move toward a random rabbit
            plan_route!(fox, rand(model.rng, map(x -> x.pos, prey)), model.landfinder)
        end
    end

    move_along_route!(fox, model, model.landfinder, model.fox_speed, model.dt)
end

function hawk_step!(hawk, model)
    # Look for rabbits nearby
    food = [x for x in nearby_agents(hawk, model) if x.type == :rabbit]
    if !isempty(food)
        # Eat (remove) the rabbit
        remove_agent!(rand(model.rng, food), model, model.airfinder)
        hawk.energy += model.Δe_rabbit
        # Fly back up
        plan_route!(hawk, hawk.pos .+ (0.0, 0.0, 7.0), model.airfinder)
    end

    # The rest of the stepping function is similar to that of foxes, except hawks use a
    # different pathfinder
    hawk.energy -= model.dt
    if hawk.energy <= 0
        remove_agent!(hawk, model, model.airfinder)
        return
    end

    rand(model.rng) <= model.hawk_repr * model.dt && reproduce!(hawk, model)

    if is_stationary(hawk, model.airfinder)
        prey = [x for x in nearby_agents(hawk, model, model.hawk_vision) if x.type == :rabbit]
        if isempty(prey)
            plan_route!(
                hawk,
                random_walkable(hawk.pos, model, model.airfinder, model.hawk_vision),
                model.airfinder,
            )
        else
            plan_route!(hawk, rand(model.rng, map(x -> x.pos, prey)), model.airfinder)
        end
    end

    move_along_route!(hawk, model, model.airfinder, model.hawk_speed, model.dt)
end

function reproduce!(animal, model)
    animal.energy = Float64(ceil(Int, animal.energy / 2))
    add_agent_pos!(Animal(nextid(model), animal.pos, v0, animal.type, animal.energy), model)
end

function model_step!(model)
    # To prevent copying of data, obtain a view of the part of the grass matrix that
    # doesn't have any grass, and grass can grow there
    growable = view(
        model.grass,
        model.grass .== 0 .& model.water_level .< model.heightmap .<= model.grass_level,
    )
    # Grass regrows with a random probability, scaling with the amount of time passing
    # each step of the model
    growable .= rand(model.rng, length(growable)) .< model.regrowth_chance * model.dt
end

function animalcolor(a)
    if a.type == :rabbit
        :brown
    elseif a.type == :fox
        :orange
    else
        :blue
    end
end

function static_preplot!(ax, model)
    surface!(
        ax,
        (100/205):(100/205):100,
        (100/205):(100/205):100,
        model.heightmap;
        colormap=:terrain
    )
end

abmvideo(
    "rabbit_fox_hawk.mp4",
    model, animal_step!, model_step!;
    figure=(resolution=(800, 700),),
    frames=300,
    framerate=15,
    ac=animalcolor,
    as=1.0,
    static_preplot!,
    title="Rabbit Fox Hawk with pathfinding"
)
