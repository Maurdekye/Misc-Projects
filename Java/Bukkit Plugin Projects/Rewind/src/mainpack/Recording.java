package mainpack;

import org.bukkit.Effect;
import org.bukkit.Location;
import org.bukkit.block.Block;
import org.bukkit.configuration.ConfigurationSection;
import org.bukkit.entity.Player;

import java.util.ArrayList;
import java.util.UUID;

public class Recording {

    String name;
    long beginstamp;
    UUID owner;

    ArrayList<Step> recordsteps;

    public Recording(String name, long beginstamp, Player owner) {
        this.name = name;
        this.beginstamp = beginstamp;
        this.owner = owner.getUniqueId();

        this.recordsteps = new ArrayList<>();
    }

    @SuppressWarnings("unused")
    public Recording(ConfigurationSection savefile, String path) {
        // TODO create deserialization
    }

    public void addStep(BlockInformation previous, BlockInformation subsequent, Location pos, long curtime) {
        recordsteps.add(new Step(previous, subsequent, pos, curtime, this.beginstamp));
    }

    @SuppressWarnings("unused")
    public void saveTo(ConfigurationSection savefile) {
        // TODO create serialization
    }

}

class Step {

    long timestamp;
    BlockInformation previous;
    BlockInformation subsequent;
    Location pos;

    public Step(BlockInformation previous, BlockInformation subsequent, Location pos, long curstamp, long basestamp) {
        this.timestamp = curstamp - basestamp;
        this.previous = previous;
        this.subsequent = subsequent;
        this.pos = pos;
    }

    @SuppressWarnings("deprecated")
    public TimeState getTS() {
        Block perspective = pos.getBlock();
        if (perspective.getType() == previous.blockType &&
                perspective.getData() == previous.blockData)
            return TimeState.FORMER;
        else if (perspective.getType() == subsequent.blockType &&
                perspective.getData() == subsequent.blockData)
            return TimeState.LATTER;
        else return TimeState.NEITHER;
    }

}

enum TimeState {
    FORMER, LATTER, NEITHER
}

class Tracer {

    Recording parent;
    long maxframe;

    long speed = 1;
    long curframe = 0;
    boolean paused = false;
    boolean rewind = false;
    boolean resetting = false;

    public Tracer(Recording parent) {
        this.parent = parent;
        this.maxframe = parent.recordsteps.get(parent.recordsteps.size()-1).timestamp;
    }

    public void advance() {
        speed = Math.abs(speed);
        if (resetting) {
            curframe = 0;
            rewind = true;
            paused = false;
        }

        if (paused) return;

        if (rewind) {
            curframe = Math.max(0, curframe - speed);
            for (Step s : parent.recordsteps) {
                if (s.timestamp >= curframe) {
                    if (s.getTS() != TimeState.FORMER)
                        s.pos.getWorld().playEffect(s.pos, Effect.STEP_SOUND, s.subsequent.blockData);
                    s.subsequent.apply(s.pos.getBlock());
                }
            }
        } else {
            curframe = Math.min(curframe + speed, maxframe);
            for (Step s : parent.recordsteps) {
                if (s.timestamp <= curframe) {
                    if (s.getTS() != TimeState.LATTER)
                        s.pos.getWorld().playEffect(s.pos, Effect.STEP_SOUND, s.subsequent.blockData);
                    s.subsequent.apply(s.pos.getBlock());
                }
            }
        }
    }

}