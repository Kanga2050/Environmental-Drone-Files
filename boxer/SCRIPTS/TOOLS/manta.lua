local done = false
local msg = "ENTER: set up Manta"

local mixes = {
  { "Roll", "ail" },
  { "Ptch", "ele" },
  { "Thr", "thr" },
  { "Yaw", "rud" },
  { "Arm", "sa" },
  { "Mode", "sb" },
  { "RTL", "sc" },
  { "Plan", "sd" },
}

local function src(n)
  local f = getFieldInfo(n)
  return f and f.id
end

local function apply()
  for ch = 0, 7 do
    local s = src(mixes[ch + 1][2])
    if not s then
      return "missing " .. mixes[ch + 1][2]
    end
  end
  model.setInfo({ name = "Manta" })
  model.deleteMixes()
  for ch = 0, 7 do
    model.insertMix(ch, 0, { name = mixes[ch + 1][1], source = src(mixes[ch + 1][2]), weight = 100, offset = 0 })
  end
  return "done: AETR + SA arm, SB mode, SC RTL, SD plan"
end

local function run(event)
  lcd.clear()
  lcd.drawText(2, 2, "Manta setup", MIDSIZE)
  lcd.drawText(2, 22, "CH5 SA arm  CH6 SB mode", SMLSIZE)
  lcd.drawText(2, 32, "CH7 SC RTL  CH8 SD plan", SMLSIZE)
  lcd.drawText(2, 50, msg, SMLSIZE + (done and 0 or INVERS))
  if not done and event == EVT_VIRTUAL_ENTER then
    msg = apply()
    done = true
    playTone(1200, 150, 0)
  end
  if event == EVT_VIRTUAL_EXIT then
    return 1
  end
  return 0
end

return { run = run }
