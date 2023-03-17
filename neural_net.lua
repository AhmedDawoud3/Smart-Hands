math.randomseed(os.time())
local exp = math.exp
local random = math.random
local InTable = function(v, table)
    for i, q in pairs(table) do
        if v == q then
            return true
        end
    end
    return false
end

local value = {}
value.__index = value

local function new(data, _children, _op, _label)
    local children = {}
    for i, child in ipairs(_children or {}) do
        if not InTable(child, children) then
            table.insert(children, child)
        end
    end
    return setmetatable({
        -- The data of the node.
        data = data,
        -- The gradient of the node.
        grad = 0,
        -- The function to call when back-propagating.
        _backward = function()
        end,
        -- The children of the node.
        _prev = children,
        -- The operation performed on the node.
        _op = _op or '',
        -- The label of the node.
        label = _label or '',
    }, value)
end

function value:__add(other)
    if type(self) == 'number' then
        return other + self
    end
    other = type(other) == 'table' and other or new(other, nil, nil, 'auto created from +')

    local out = new(self.data + other.data, {self, other}, '+')

    local _backward = function()
        self.grad = self.grad + out.grad
        other.grad = other.grad + out.grad
    end

    out._backward = _backward

    return out
end

function value:__mul(other)
    if type(self) == 'number' then
        return other * self
    end

    other = type(other) == 'table' and other or new(other, nil, nil, 'auto created from *')

    local out = new(self.data * other.data, {self, other}, '*')

    local _backward = function()
        self.grad = self.grad + other.data * out.grad
        other.grad = other.grad + self.data * out.grad
    end
    out._backward = _backward

    return out
end

function value:sin()
    local out = new(math.sin(self.data), {self}, "sin")

    local _backward = function()
        self.grad = self.grad + math.cos(self.data) * out.grad
    end
    out._backward = _backward

    return out
end

function value:cos()
    local out = new(math.cos(self.data), {self}, "cos")

    local _backward = function()
        self.grad = self.grad - math.sin(self.data) * out.grad
    end
    out._backward = _backward

    return out
end


function value:__pow(other)
    local out = new(self.data ^ other, {self}, "^" .. other)

    local _backward = function()
        self.grad = self.grad + other * (self.data ^ (other - 1)) * out.grad
    end
    out._backward = _backward

    return out
end

function value:GetTopo()
    local build_topo

    local topo = {} -- topological order
    local visited = {}

    build_topo = function(v)
        if not InTable(v, visited) then
            table.insert(visited, v)
            for _, child in pairs(v._prev) do
                build_topo(child)
            end
            table.insert(topo, v)
        end
    end

    build_topo(self)

    return topo
end

function value:backward()
    local build_topo

    local topo = {} -- topological order
    local visited = {}

    build_topo = function(v)
        if not InTable(v, visited) then
            table.insert(visited, v)
            for _, child in pairs(v._prev) do
                build_topo(child)
            end
            table.insert(topo, v)
        end
    end

    build_topo(self)

    self.grad = 1
    -- Loop in reverse order
    for i = #topo, 1, -1 do
        topo[i]:_backward()
    end

end

function value:exp()
    local out = new(exp(self.data), {self}, "e^")

    local _backward = function()
        self.grad = self.grad + self.data * out.grad
    end
    out._backward = _backward

    return out
end

function value:tanh()

    local out = new((exp(self.data * 2) - 1) / (exp(self.data * 2) + 1), {self}, 'tanh')

    local _backward = function()
        self.grad = self.grad + (1 - out.data ^ 2) * out.grad
    end
    out._backward = _backward

    return out
end

function value:relu()

    -- If the data is less than zero, return zero, otherwise return the data.
    local out = new(math.max(0, self.data), {self}, 'ReLU')

    -- If the data is greater than zero, then the gradient is 1 * the gradient of the output.
    local _backward = function()
        self.grad = self.grad + ((out.data > 0) and 1 or 0) * out.grad
    end
    out._backward = _backward

    return out
end

function value:segmoid()

    -- Segmoid function
    local out = new(1 / (1 + exp(-self.data)), {self}, 'Segmoid')

    -- Back propagate
    local _backward = function()
        self.grad = self.grad + (out.data * (1 - out.data)) * out.grad
    end
    out._backward = _backward

    return out
end

function value:__div(other)
    return self * other ^ -1
end

function value:__sub(other)
    return self + (other * -1)
end

function value:__concat(other)
    return other .. tostring(self)
end

function value:__tostring()
    if self.label ~= "" then
        return "Value(data=" .. self.data .. ", grad=" .. self.grad .. ")" .. "   [" .. self.label .. "]"
    end
    return "Value(data=" .. self.data .. ", grad=" .. self.grad .. ")"
end

-- the module
Value = setmetatable({
    new = new,
}, {
    __call = function(_, ...)
        return new(...)
    end,
})

-- Neuron class
local neuron = {}
neuron.__index = neuron

local function new(nin)
    -- Start with random weights
    local w = {}
    for i = 1, nin do
        -- make the data as a random float from -1 to 1
        w[i] = Value((random() - .5) * 2, nil, nil, 'w')
    end

    -- Start with a bais of 0
    local b = Value(0, nil, nil, 'b')

    return setmetatable({
        w = w,
        b = b,
    }, neuron)
end

ACTIVATIONS = {
    tanh = 'tanh',
    relu = 'relu',
    segmoid = 'segmoid',
}

function neuron:__call(x, activation)
    local sum = 0 + self.b -- initialize the sum to the bais

    for i = 1, #self.w do
        sum = sum + self.w[i] * x[i] -- add the product of the weight and input to the sum
    end

    if not activation then
        return sum:tanh() -- return the tanh of the sum
    elseif activation == ACTIVATIONS.tanh then
        return sum:tanh() -- return the tanh of the sum
    elseif activation == ACTIVATIONS.relu then
        return sum:relu() -- return the relu of the sum
    elseif activation == ACTIVATIONS.segmoid then
        return sum:segmoid() -- return the segmoid of the sum
    end

end

function neuron:printParameters()
    for i, v in ipairs(self:parameters()) do
        print(v)
    end
end

function neuron:parameters()
    local T = {self.b}
    for i = 1, #self.w do
        T[#T + 1] = self.w[i]
    end
    return T
end

function neuron:__tostring()
    return "Neuron(" .. #self.w .. ")"
end

Neuron = setmetatable({
    new = new,
}, {
    __call = function(_, ...)
        return new(...)
    end,
})

local layer = {}
layer.__index = layer

local function new(numInputs, numOutputs)
    -- Create a table to hold the neurons
    local neurons = {}

    -- Loop through the number of neurons to create
    for i = 1, numOutputs do
        -- Create a new neuron with the number of inputs
        neurons[i] = Neuron(numInputs)
    end

    -- Return a layer object with the created neurons
    return setmetatable({
        neurons = neurons,
    }, layer)
end

function layer:__call(x)
    --[[
        This function calls the layer, which calls the neurons within the layer.
    ]] --

    local out = {}
    for i = 1, #self.neurons do
        out[i] = self.neurons[i](x, self.activation)
    end

    if #out > 1 then
        return out
    else
        return out[1]
    end
end

function layer:printParameters()
    for _, neuron in ipairs(self.neurons) do
        neuron:printParameters()
    end
end

function layer:parameters()
    local T = {}
    for i = 1, #self.neurons do
        local pr = self.neurons[i]:parameters()
        for _, parameter in ipairs(pr) do
            T[#T + 1] = parameter
        end
    end
    return T
end

function layer:__tostring()
    local str = ''
    for i = 1, #self.neurons do
        str = str .. tostring(self.neurons[i]) .. ', '
    end
    str = str:sub(1, -3)
    return "Layer of[" .. str .. "]"
end

Layer = setmetatable({
    new = new,
}, {
    __call = function(_, ...)
        return new(...)
    end,
})

local network = {}
network.__index = network

local function new(nin, nouts)
    -- Start with nin as it's the input layer
    local layer_sizes = {nin}
    for i = 1, #nouts do
        layer_sizes[#layer_sizes + 1] = nouts[i]
    end

    local layers = {}
    for i = 1, #nouts do
        layers[#layers + 1] = Layer(layer_sizes[i], layer_sizes[i + 1])
    end
    return setmetatable({
        layers = layers,
    }, network)
end

function network:__call(x)
    -- It's called so that the last layer's output is returned
    for _, layer in ipairs(self.layers) do
        x = layer(x)
    end
    return x
end

function network:__tostring()
    local str = ''
    for i = 1, #self.layers do
        str = str .. tostring(self.layers[i]) .. '\n'
    end
    -- str = str:sub(1, -3)
    return "MLP of[\n" .. str .. "]"
end

function network:printParameters()
    for _, layer in ipairs(self.layers) do
        layer:printParameters()
    end
end

function network:parameters()
    local T = {}
    for i = 1, #self.layers do
        local pr = self.layers[i]:parameters()
        for _, parameter in ipairs(pr) do
            T[#T + 1] = parameter
        end
    end
    return T
end

Network = setmetatable({
    new = new,
}, {
    __call = function(_, ...)
        return new(...)
    end,
})
