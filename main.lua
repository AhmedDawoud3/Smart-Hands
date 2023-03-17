Class = require 'lib/class'
require 'lib/color'
local debugGraph = require 'lib.debugGraph'
require("robot")
require("neural_net")

BackgroundColor = GRAY_E
Window = {
    width = 1280,
    height = 720,
}

local robot = Robot(Window.width / 2, 600, {800, 300})
local lossGrapp = debugGraph:new('custom', 105, 5, 400, 60, 1 / 60, '', love.graphics.newFont(16))

function love.load()
    love.window.setTitle("Smart Robot Hands")
    love.window.setMode(Window.width, Window.height)
end

function love.update(dt)
    if love.keyboard.isDown('escape') then
        love.event.quit()
    end
    Mouse = {love.mouse.getPosition()}

    robot:update(dt)

    lossGrapp:update(dt, robot.loss.data)
end

function love.draw()
    BackgroundColor:SetBackground()

    GOLD:Set()
    lossGrapp:draw()

    GRAY_A:Set()
    love.graphics.print('[Loss]: ' .. NumberGSUB(robot.loss.data, 4), 110, 7)
    love.graphics.rectangle('line', 100, 0, 410, 70)

    robot:draw()

    RED:Set()
    love.graphics.setPointSize(10)
    love.graphics.points(Mouse)

    -- FPS
    GREEN:Set()
    love.graphics.print("FPS: " .. love.timer.getFPS(), 10, 10)
end

---@param t string|integer number
---@param n integer number of decimal places
---@return string number formated
function NumberGSUB(t, n)
    local time = tonumber(t)
    local seconds = math.floor(time)
    local miliseconds = time - seconds
    local formated = tostring(seconds) .. string.sub(tostring(miliseconds), 2, n + 2)
    return formated
end
