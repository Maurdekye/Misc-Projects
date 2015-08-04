package mainpack;


import org.bukkit.Material;
import org.bukkit.block.Block;
import org.bukkit.block.Sign;

import java.util.Arrays;
import java.util.List;

class BlockInformation {

    Material blockType;
    byte blockData;
    List<String> signText;

    @SuppressWarnings("deprecated")
    public BlockInformation(Block input) {
        blockType = input.getType();
        blockData = input.getData();
        if (input.getState() instanceof Sign)
            signText = Arrays.asList(((Sign) input.getState()).getLines());
        // TODO gather relevant block data
    }

    @SuppressWarnings("deprecated")
    public void apply(Block block) {
        if (block.getType() != blockType)
            block.setType(this.blockType);
        if (block.getData() != blockData)
            block.setData(this.blockData);
        if (block.getState() instanceof Sign)
            for (int i=0;i<signText.size();i++) ((Sign) block.getState()).setLine(i, signText.get(i));
        // TODO set relevant block data
    }
}
