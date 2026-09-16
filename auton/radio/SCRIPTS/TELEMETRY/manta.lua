local last = 0
local sent = 0

local function num(n)
  local v = getValue(n)
  if type(v) == "number" then
    return v
  end
  return 0
end

local function txt(n)
  local v = getValue(n)
  if type(v) == "string" then
    return v
  end
  return "--"
end

local function bg()
  local p = getValue("sd")
  if p > 0 and last <= 0 then
    sent = getTime()
    playHaptic(60, 20, 0)
    playTone(1500, 120, 0)
  end
  last = p
  if num("RQly") > 0 and num("RQly") < 50 then
    playHaptic(20, 200, 0)
  end
end

local function run(event)
  bg()
  local rq, sats, v, pct, alt = num("RQly"), num("Sats"), num("RxBt"), num("Bat%"), num("Alt")
  lcd.clear()
  lcd.drawText(1, 1, "MANTA", MIDSIZE)
  lcd.drawText(48, 4, txt("FM"), INVERS)
  lcd.drawText(1, 18, string.format("LQ %3d%%", math.floor(rq)), rq < 70 and BLINK or 0)
  lcd.drawText(64, 18, string.format("Sats %2d", math.floor(sats)), sats < 8 and BLINK or 0)
  lcd.drawText(1, 28, string.format("Bat %4.1fV %3d%%", v, math.floor(pct)), (pct > 0 and pct < 30) and BLINK or 0)
  lcd.drawText(1, 38, string.format("Alt %4.0fm", alt), 0)
  local p = getValue("sd") > 0
  local flag = p and INVERS or 0
  if p and getTime() - sent < 300 then
    flag = flag + BLINK
  end
  lcd.drawText(1, 52, p and "SD: SEND SURVEY" or "SD: survey held", flag)
  return 0
end

return { run = run, background = bg }
