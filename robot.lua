Robot = Class:extend("Robot")
local Hand = Class:extend("Hand")
local handClawImage = love.graphics.newImage("hand.png")

function Hand:new(x, y, angle)
    self.x = x
    self.x.label = "x hand"
    self.y = y
    self.y.label = "y hand"
    self.w = 7
    self.h = 120
    self.angle = angle or 0
    self.angle = Value(self.angle, nil, nil, 'angle')
end

function Hand:draw(lastHand)
    if lastHand then
        local x, y = self.x.data, self.y.data
        local angle = self.angle.data
        GRAY_B:Set()
        love.graphics.line(x, y, x + self.h * 0.8 * math.cos(angle), y - self.h * 0.8 * math.sin(angle))
        Color.Reset()
        love.graphics.draw(handClawImage, x + self.h * 0.8 * math.cos(angle), y - self.h * 0.8 * math.sin(angle),
                           math.pi / 2 - angle, 1, 1, self.h * 0.1, self.h * 0.2)
        return
    end

    local angle = self.angle.data
    local x, y = self.x.data, self.y.data
    GRAY_B:Set()
    love.graphics.setLineWidth(self.w)
    love.graphics.line(x, y, x + self.h * math.cos(angle), y - self.h * math.sin(angle))
    TEAL:Set()
    love.graphics.setPointSize(5)
    love.graphics.circle("fill", x, y, 5)
    love.graphics.circle("fill", x + self.h * math.cos(angle), y - self.h * math.sin(angle), 5)
end

function Robot:new(x, y)
    self.x = x
    self.y = y

    self.hands = {}
    for i = 1, 4 do
        self.InsertHand(self, math.pi + i * math.pi / 8)
    end
end

function Robot:update(dt)
    self.target = {
        x = Mouse[1],
        y = Mouse[2],
    }
    for i = 2, #self.hands do
        self.hands[i].x = self.hands[i - 1].x + self.hands[i - 1].h * self.hands[i - 1].angle:cos()
        self.hands[i].y = self.hands[i - 1].y - self.hands[i - 1].h * self.hands[i - 1].angle:sin()
    end

    local lastHand = self.hands[#self.hands]

    -- Zero Grad for all the values
    for i, v in ipairs(self.hands) do
        v.angle.grad = 0
    end

    -- Calculate the loss
    self.loss = (self.target.x - (lastHand.x + lastHand.angle:cos() * lastHand.h)) ^ 2
                    + (self.target.y - (lastHand.y - lastHand.angle:sin() * lastHand.h)) ^ 2
    self.loss:backward()

    -- Check if more than two hands have the same grad for the angle then change each by a random value
    local sameGrad = {}
    for i, v in ipairs(self.hands) do
        if sameGrad[v.angle.grad] then
            sameGrad[v.angle.grad] = sameGrad[v.angle.grad] + 1
        else
            sameGrad[v.angle.grad] = 1
        end
    end
    for i, v in ipairs(self.hands) do
        if sameGrad[v.angle.grad] > 1 then
            v.angle.grad = v.angle.grad + love.math.random(-1, 1)
        end
    end

    for _, v in ipairs(self.hands) do
        v.angle.data = v.angle.data - v.angle.grad / 10e4 * 0.5
    end
end

function Robot:draw()
    for i = 1, #self.hands do
        self.hands[i]:draw(i == #self.hands)
    end

    YELLOW:Set()
    love.graphics.setLineWidth(1)
    local lastHand = self.hands[#self.hands]
    local x, y = lastHand.x.data, lastHand.y.data
    local angle, h = lastHand.angle.data, lastHand.h
    love.graphics.line(self.target.x, self.target.y, x + math.cos(angle) * h, y - math.sin(angle) * h)

end

function Robot:InsertHand(angle)
    if #self.hands == 0 then
        table.insert(self.hands, Hand(Value(self.x), Value(self.y), angle))
        return
    end
    local lastHand = self.hands[#self.hands]
    table.insert(self.hands, Hand(lastHand.x + lastHand.h * lastHand.angle:cos(),
                                  lastHand.y - lastHand.h * lastHand.angle:sin(), angle))
end

